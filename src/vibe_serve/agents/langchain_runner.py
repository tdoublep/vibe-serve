"""LangChain ``create_agent`` implementation of :class:`AgentRunner`.

This backend replaces :class:`DeepAgentsRunner` for models where the
``deepagents`` + ``AutoStrategy`` stack stalls. The root cause:

- ``langchain.agents.create_agent(response_format=AutoStrategy(Cls))``
  inspects ``_supports_provider_strategy(model)``. For ``claude-opus-4-7``
  this returns ``False``, so ``AutoStrategy`` falls back to
  ``ToolStrategy``.
- ``ToolStrategy`` binds the model with ``tool_choice="any"``
  (factory.py:1242 hardcodes it whenever any structured-output tool is
  registered). That forces *some* tool call per turn but never the
  *finalize* tool — the schema tool is just one of nine options.
- ``claude-opus-4-7`` keeps picking ``ls``/``read_file`` indefinitely
  instead of finalizing, so the loop stalls before the orchestrator can
  emit one ``OrchestratorPlan``.

This runner sidesteps the issue by:

1. Owning the system prompt outright — no concatenation with the
   ``deepagents`` ``BASE_AGENT_PROMPT`` boilerplate.
2. Skipping ``response_format`` on the main agent. The orchestrator /
   implementer / judge prompts already tell the model to return JSON, so
   we let it answer in plain text and parse the JSON via the existing
   :func:`vibe_serve.agent_runner._parse_typed_response_text` helper.
3. If JSON extraction from the final AI message fails, we run a single
   forced-tool-call extraction turn that binds the model to *one* tool
   (the schema tool) with ``tool_choice=response_cls.__name__``. Each
   provider integration translates the string form to its native shape
   (Anthropic ``{"type":"tool","name":...}`` / OpenAI
   ``{"type":"function","function":{"name":...}}``), so this works
   against both Anthropic and an OpenAI-compatible endpoint like vLLM.
4. Reusing the ``deepagents`` ``FilesystemMiddleware`` for the
   ``ls``/``read_file``/``write_file``/``edit_file``/``glob``/``grep``/
   ``execute`` tool set, but dropping ``TodoListMiddleware`` and
   ``SubAgentMiddleware`` — both are noise for orchestrator/implementer/
   judge.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError
from vibe_serve._agent_cli.base import MCPServerSpec

from vibe_serve.agent_runner import (
    _DEFAULT_MAX_TEXT_LEN,
    _extract_last_ai_message_text,
    _extract_todos,
    _log_agent_config,
    _log_and_print,
    _parse_typed_response_text,
)
from vibe_serve.agents.callbacks import AgentLogger, TodoDisplay
from vibe_serve.guardrails import AgentGuardrailsMiddleware

T = TypeVar("T", bound=BaseModel)


def _agent_label(kind: str) -> str:
    """Convert ``"perf_eval"`` to ``"Perf Eval"``, etc."""
    return kind.replace("_", " ").title()


def _build_finalize_tool(response_cls: type[T]) -> StructuredTool:
    """Build a single-purpose tool whose args match ``response_cls``."""
    return StructuredTool(
        name=response_cls.__name__,
        description=(
            f"Submit the final {response_cls.__name__} response. "
            "Call this exactly once when you are done. "
            "All your reasoning should already be reflected in the arguments."
        ),
        args_schema=response_cls,
        func=lambda **kwargs: kwargs,
    )


def _extract_via_forced_tool_call(
    *,
    model: Any,
    response_cls: type[T],
    system_prompt: str,
    user_prompt: str,
    last_ai_message: str,
    log_file,
    label: str,
    round_label: str,
) -> T | None:
    """Run one model turn with ``tool_choice`` forced to the schema tool.

    Used as a fallback when the agent's last AI message did not contain a
    parseable JSON object matching ``response_cls``. Passing the tool
    name as a string lets each provider integration emit its native
    forced-tool shape, guaranteeing the next message is a tool call to
    that specific tool — no agent loop, one round-trip.
    """
    finalize_tool = _build_finalize_tool(response_cls)
    # Pass the tool name as a plain string. Both langchain_anthropic and
    # langchain_openai translate `tool_choice="<name>"` to their provider's
    # forced-tool shape — Anthropic's {"type":"tool","name":...} and
    # OpenAI's {"type":"function","function":{"name":...}}. The dict form
    # of either provider is not portable: vLLM rejects the Anthropic shape
    # with HTTP 400 (verified session 18).
    bound = model.bind_tools(
        [finalize_tool],
        tool_choice=response_cls.__name__,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    if last_ai_message:
        messages.append(AIMessage(content=last_ai_message))
    messages.append(
        HumanMessage(
            content=(
                f"Now call the {response_cls.__name__} tool once with the "
                "final structured payload. Do not call any other tool."
            )
        )
    )

    if log_file:
        log_file.write(
            f"\n=== {label} FORCED-TOOL EXTRACTION: {round_label} ===\n"
            f"  bound tool: {response_cls.__name__}\n"
        )
        log_file.flush()

    try:
        result = bound.invoke(messages)
    except Exception as exc:  # noqa: BLE001 — telemetry path; we return None
        if log_file:
            log_file.write(f"  forced-tool extraction failed: {type(exc).__name__}: {exc}\n")
            log_file.flush()
        return None

    tool_calls = getattr(result, "tool_calls", None) or []
    for call in tool_calls:
        if call.get("name") != response_cls.__name__:
            continue
        args = call.get("args") or {}
        try:
            return response_cls.model_validate(args)
        except ValidationError as exc:
            if log_file:
                log_file.write(f"  forced-tool args failed validation: {exc}\n")
                log_file.flush()
            continue

    # Last-ditch: parse JSON from the assistant text, if any.
    content = getattr(result, "content", "")
    text = content if isinstance(content, str) else json.dumps(content)
    return _parse_typed_response_text(text, response_cls)


class LangChainAgentRunner:
    """:class:`AgentRunner` backed by :func:`langchain.agents.create_agent`.

    The constructor mirrors :class:`DeepAgentsRunner` so it slots straight
    into :func:`vibe_serve.agents.build_agent_runner`.
    """

    backend_name = "langchain"

    def __init__(
        self,
        *,
        model: Any,
        backends: dict[str, Any],
        skills: list[str],  # noqa: ARG002 — kept for signature parity; this runner does not surface skills as a separate middleware
        model_name: str | None,
        run_log_file,
    ):
        self._model = model
        self._backends = backends
        self._model_name = model_name
        self._run_log_file = run_log_file

    def invoke(
        self,
        *,
        kind: str,
        workspace: Path,  # noqa: ARG002 — backend already encapsulates cwd
        system_prompt: str,
        env: dict[str, str] | None = None,  # noqa: ARG002 — env lives on the BaseSandbox
        user_prompt: str,
        response_cls: type[T],
        fallback_factory: Callable[[], T],
        round_label: str,
        mcp_servers: list[MCPServerSpec] | None = None,  # noqa: ARG002 — cli-only injection point
        tools: list[BaseTool] | None = None,
    ) -> T:
        label = _agent_label(kind)

        # Fresh checkpointer + thread id per invocation. Same pattern as
        # DeepAgentsRunner: every phase starts with a clean context window.
        checkpointer = MemorySaver()
        thread_id = uuid.uuid4().hex

        backend = self._backends[kind]
        extra_tools = list(tools or [])

        # Middleware stack — explicitly NOT TodoListMiddleware (write_todos
        # is noise for orchestrator/implementer/judge) and NOT
        # SubAgentMiddleware (the `task` tool is noise too). We keep
        # filesystem tools, prompt caching for Anthropic, and our own
        # guardrails.
        middleware = [
            FilesystemMiddleware(backend=backend),
            AgentGuardrailsMiddleware(backend=backend),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        ]

        # Critical: own the system prompt outright. No concatenation with
        # deepagents' BASE_AGENT_PROMPT.
        agent = create_agent(
            self._model,
            tools=extra_tools,
            system_prompt=system_prompt,
            middleware=middleware,
            checkpointer=checkpointer,
        )
        _log_agent_config(agent, label, self._run_log_file)

        callbacks = [
            AgentLogger(
                log_file=self._run_log_file,
                model_name=self._model_name,
                agent_label=label,
            )
        ]

        # Phase 1: run the agent with tools available. No response_format
        # — the prompts already ask for JSON.
        last_ai_message = self._stream_agent(
            agent=agent,
            prompt=user_prompt,
            response_cls=response_cls,
            label=kind.upper(),
            callbacks=callbacks,
            thread_id=thread_id,
            round_label=round_label,
        )

        # Try parsing the final assistant message as the structured
        # response. The prompts already ask for raw JSON, so this is the
        # happy path.
        parsed = _parse_typed_response_text(last_ai_message, response_cls)
        if parsed is not None:
            output_json = parsed.model_dump_json(indent=2)
            _log_and_print(
                f"\n=== {label.upper()} ROUND OUTPUT ===",
                self._run_log_file,
            )
            _log_and_print(output_json, self._run_log_file, max_len=_DEFAULT_MAX_TEXT_LEN)
            return parsed

        # Phase 2 fallback: force a single bound-tool call to extract the
        # structured payload. This is the fix path the findings doc
        # describes — guarantees finalization in one round-trip.
        _log_and_print(
            f"\n=== {label.upper()} ROUND OUTPUT (no JSON in last AI message — forcing tool) ===",
            self._run_log_file,
        )
        if last_ai_message:
            _log_and_print(last_ai_message, self._run_log_file, max_len=_DEFAULT_MAX_TEXT_LEN)
        forced = _extract_via_forced_tool_call(
            model=self._model,
            response_cls=response_cls,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            last_ai_message=last_ai_message,
            log_file=self._run_log_file,
            label=label.upper(),
            round_label=round_label,
        )
        if forced is not None:
            output_json = forced.model_dump_json(indent=2)
            _log_and_print(
                f"\n=== {label.upper()} FORCED-TOOL OUTPUT ===",
                self._run_log_file,
            )
            _log_and_print(output_json, self._run_log_file, max_len=_DEFAULT_MAX_TEXT_LEN)
            return forced

        _log_and_print(
            f"\n=== {label.upper()} ROUND OUTPUT (fallback) ===",
            self._run_log_file,
        )
        _log_and_print(
            f"Forced-tool extraction also failed; returning fallback {response_cls.__name__}.",
            self._run_log_file,
        )
        return fallback_factory()

    def _stream_agent(
        self,
        *,
        agent,
        prompt: str,
        response_cls: type[T],  # noqa: ARG002 — reserved for future structured_response support
        label: str,
        callbacks: list,
        thread_id: str,
        round_label: str,
    ) -> str:
        """Run the agent's stream and return the final AI message text."""
        todo_display = TodoDisplay()
        callbacks_label = " + ".join(type(cb).__name__ for cb in callbacks)
        last_ai_message = ""
        log_file = self._run_log_file
        _log_and_print(f"\n=== {label} ROUND START: {round_label} ===", log_file)
        _log_and_print(f"callbacks: {callbacks_label}", log_file)
        _log_and_print(f"thread_id: {thread_id}", log_file)
        _log_and_print("--- input ---", log_file)
        _log_and_print(prompt, log_file, max_len=_DEFAULT_MAX_TEXT_LEN)
        try:
            for update in agent.stream(
                {"messages": [("human", prompt)]},
                config={"callbacks": callbacks, "configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                todos = _extract_todos(update)
                if todos is not None:
                    todo_display.update(todos)
                extracted_text = _extract_last_ai_message_text(update)
                if extracted_text:
                    last_ai_message = extracted_text
        except Exception as exc:
            error_text = f"error: {type(exc).__name__}: {exc}"
            _log_and_print(f"\n=== {label} ROUND ERROR: {round_label} ===", log_file)
            _log_and_print(error_text, log_file, max_len=_DEFAULT_MAX_TEXT_LEN)
            raise
        return last_ai_message

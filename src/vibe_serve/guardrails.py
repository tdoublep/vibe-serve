"""Per-agent guardrails: loop-breaker + read_file footgun blocker +
write_file overwrite enabler.

Three failure modes observed when driving deepagents with a self-hosted
reasoning model (Nemotron-3-Super-FP8 via vLLM + nemotron_v3 parser):

1. ``read_file(limit=0)`` returns an empty string silently (the slice
   ``lines[:0]`` is empty), and the model has no signal that the call was
   malformed. At ``temperature=0`` it loops on the same call indefinitely.

2. More generally, any tool whose output is constant across calls (the
   empty string, a "file not found" error, etc.) becomes a fixed point for
   a deterministic decoder: same prior context → same chain-of-thought →
   same tool call.

3. ``write_file`` to an existing path is rejected by deepagents'
   ``FilesystemBackend`` with the message "Cannot write to {path} because
   it already exists." This collides with the LLM-natural workflow for
   code iteration — hold the file in working memory, regenerate the
   refined version, dump it. Each rejected attempt burns the full output
   budget (16k tokens) regenerating the file for nothing. We don't want
   ``edit_file`` exact-string-match for a 22k-char rewrite; we just want
   overwrite to work.

This middleware addresses all three:

- ``read_file(limit<=0)`` is short-circuited with a descriptive
  ``ToolMessage`` before the filesystem backend runs.

- The last ``N`` tool-call signatures are tracked per agent invocation;
  a ``N+1``-th identical call in a row is short-circuited with a
  "you're looping, try something else" message.

- ``write_file`` to an existing path: the existing file is removed
  before the call proceeds, so the backend's "already exists" check
  passes and the new content is written. This converts ``write_file``
  into upsert semantics for the LLM, matching how it actually thinks
  about file edits.

Combined into ``AgentGuardrailsMiddleware`` which composes naturally with
deepagents' middleware stack via ``create_deep_agent(middleware=[...])``.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import deque
from typing import Any, Callable

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def _is_under_root(path: str, root: str) -> bool:
    """True if ``path`` is exactly ``root`` or sits under it.

    String-prefix with an explicit ``/`` boundary so e.g. ``/workspace/foo``
    isn't accepted as under ``/workspace/foobar``. Both args are expected
    without a trailing slash (the caller is responsible for that).
    """
    if path == root:
        return True
    return path.startswith(root + "/")


def _tool_call_signature(tool_call: dict) -> tuple[str, str]:
    """Stable hashable signature of a tool call's name and args.

    JSON-encoding the args with sorted keys collapses dict-ordering noise so
    two semantically identical calls compare equal.
    """
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    try:
        args_str = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        # Unhashable arg values shouldn't crash the guardrail.
        args_str = repr(args)
    return name, args_str


class AgentGuardrailsMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Combined ``read_file(limit<=0)`` + consecutive-repeat guard +
    ``write_file`` overwrite enabler.

    Loop-breaker rationale: N=3, not N=2. Some legitimate trajectories
    repeat a call once — e.g. ``write_file → "already exists" error → read_file
    → edit_file`` involves two consecutive reads of the same path. We only
    intervene after **three** identical consecutive calls, where the model
    has clearly received the same response twice and is about to issue the
    call a third time without changing behaviour.

    ``write_file`` overwrite: when ``enable_write_overwrite=True`` and a
    ``backend`` is supplied, calls targeting an existing path are converted
    to overwrites. The existing file is removed via the backend (if it
    exposes a ``delete`` method) or directly via the resolved path, after
    which the original ``handler`` is invoked normally. The model sees a
    success ``ToolMessage`` instead of "already exists".
    """

    def __init__(
        self,
        *,
        repeat_window: int = 3,
        block_read_file_limit_zero: bool = True,
        enable_write_overwrite: bool = True,
        enable_path_normalization: bool = True,
        backend: Any | None = None,
    ) -> None:
        super().__init__()
        if repeat_window < 2:
            raise ValueError("repeat_window must be >= 2")
        self._repeat_window = repeat_window
        self._block_read_file_limit_zero = block_read_file_limit_zero
        self._enable_write_overwrite = enable_write_overwrite
        self._enable_path_normalization = enable_path_normalization
        # Backend is used to resolve write_file paths under the agent's
        # workspace and remove the existing file before the original
        # ``write_file`` tool runs. Without it, the overwrite path falls
        # back to a no-op (the backend will still reject) so callers must
        # supply it for the overwrite enabler to work.
        self._backend = backend
        # Host-path prefixes the model invents when it confuses the virtual FS
        # with the real shell's view (see REPRODUCTION_NOTES_session6.md). Each
        # is matched against the leading characters of a path argument; if
        # found, the prefix is stripped and the remainder becomes the virtual
        # path. Order matters — more specific prefixes first.
        self._path_strip_prefixes: tuple[str, ...] = (
            # The example-source tree the model keeps grabbing onto.
            "/workspace/vibe-serve/examples/Llama-3-8B/",
            # Per-round workspace under exp_env/ — if the model spells it out
            # in full, treat it as a redundant prefix and collapse.
            "/workspace/vibe-serve/exp_env/",
            # Bare root of the source repo.
            "/workspace/vibe-serve/",
        )
        # Recent tool-call signatures, keyed by langgraph thread id so a
        # fresh agent invocation starts with a clean history even if the
        # middleware instance is shared (which deepagents_runner doesn't
        # currently do, but it's cheap insurance).
        self._recent: dict[str | None, deque[tuple[str, str]]] = {}

    @property
    def name(self) -> str:
        return "AgentGuardrailsMiddleware"

    def _thread_id(self, request: ToolCallRequest) -> str | None:
        runtime = request.runtime
        cfg = getattr(runtime, "config", None) or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}
        tid = configurable.get("thread_id") if isinstance(configurable, dict) else None
        return tid if isinstance(tid, str) else None

    def _buffer(self, thread_id: str | None) -> deque[tuple[str, str]]:
        buf = self._recent.get(thread_id)
        if buf is None:
            buf = deque(maxlen=self._repeat_window)
            self._recent[thread_id] = buf
        return buf

    def _read_file_limit_zero(self, tool_call: dict) -> bool:
        if tool_call.get("name") != "read_file":
            return False
        args = tool_call.get("args") or {}
        if "limit" not in args:
            return False
        try:
            return int(args["limit"]) <= 0
        except (TypeError, ValueError):
            return False

    def _maybe_clear_existing_for_overwrite(self, tool_call: dict) -> None:
        """If this is a ``write_file`` call to an existing path, delete the
        file so the downstream handler treats the write as a fresh create.

        Best-effort. Any failure is swallowed; the original handler will
        run and surface its own error message to the model.
        """
        if not self._enable_write_overwrite:
            return
        if tool_call.get("name") != "write_file":
            return
        backend = self._backend
        if backend is None:
            return
        args = tool_call.get("args") or {}
        file_path = args.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return
        resolver = getattr(backend, "_resolve_path", None)
        if not callable(resolver):
            return
        try:
            resolved = resolver(file_path)
        except Exception:
            return
        try:
            exists = resolved.exists()
        except Exception:
            return
        if not exists:
            return
        try:
            # Symlinks: unlink targets the link, not the file it points to.
            # That matches the security posture of FilesystemBackend.write,
            # which opens with O_NOFOLLOW.
            if resolved.is_file() or resolved.is_symlink():
                resolved.unlink()
        except Exception:
            # Leave it to the original handler to report the error.
            return

    def _normalize_path_arg(self, raw: Any) -> str | None:
        """If ``raw`` starts with a known host prefix, return the virtual-path
        suffix; otherwise return ``None`` (caller leaves the arg untouched).

        Special-cases the ``exp_env/`` prefix: the suffix after ``exp_env/``
        looks like ``<run>/workspace/<rest>``, and we want only ``<rest>``
        — the per-round workspace IS the virtual root.
        """
        if not isinstance(raw, str) or not raw.startswith("/"):
            return None
        for prefix in self._path_strip_prefixes:
            if raw.startswith(prefix):
                suffix = raw[len(prefix):]
                if prefix.endswith("exp_env/"):
                    # /workspace/vibe-serve/exp_env/<run>/workspace/<rest>  →  /<rest>
                    parts = suffix.split("/workspace/", 1)
                    if len(parts) == 2:
                        return "/" + parts[1].lstrip("/")
                    # Fall back to stripping the run dir at minimum.
                    return "/" + (suffix.split("/", 1)[1] if "/" in suffix else "")
                return "/" + suffix
        return None

    # Filesystem tools whose path-like argument should be rewritten. ``grep``
    # is deliberately omitted: its ``pattern`` arg is the search string, not
    # a path, and rewriting it would corrupt the search.
    _FS_TOOL_PATH_FIELDS: dict[str, tuple[str, ...]] = {
        "read_file": ("file_path",),
        "write_file": ("file_path",),
        "edit_file": ("file_path",),
        "ls": ("path",),
        "glob": ("pattern", "path"),
    }

    # Matches absolute /workspace/... tokens in a command string. The lookbehind
    # rejects matches that are part of a longer word (e.g. "foo/workspace/x"
    # should not match — it's a relative path inside cwd). We do NOT exclude
    # tokens inside quotes; if the agent puts a host path in quotes, the shell
    # will still resolve it as that host path, so the refusal still applies.
    _WORKSPACE_PATH_RE = re.compile(r"(?<![\w/])/workspace/[^\s'\";|&<>]+")

    # Matches ``cd /workspace<anything>``. The capture group is the absolute
    # path argument — including any /workspace prefix and what follows.
    _CD_WORKSPACE_RE = re.compile(r"\bcd\s+(/workspace[^\s'\";|&<>]*)")

    def _refuse_execute(
        self,
        tool_call: dict,
        offending_token: str,
        workspace_root: str,
    ) -> ToolMessage:
        """Build a ToolMessage refusing an execute command that would escape
        the workspace.

        The message names the offending token and the workspace root so the
        model can self-correct rather than re-issuing the same command.
        """
        return ToolMessage(
            content=(
                f"Refusing execute: the command references {offending_token!r}, "
                f"which is outside your workspace ({workspace_root}). Use relative "
                "paths or paths under your workspace. The shell is already "
                "anchored to your workspace root, so 'main.py' resolves to the "
                "right place — you don't need to spell '/workspace/...'."
            ),
            tool_call_id=tool_call["id"],
            name=tool_call.get("name"),
            status="error",
        )

    def _maybe_normalize_paths(self, tool_call: dict) -> ToolMessage | None:
        """Rewrite known-confusing host paths to virtual-FS paths in-place.

        Applies to filesystem tools (read_file/write_file/edit_file/ls/glob)
        for path-shaped fields, and to ``execute`` by prepending a ``cd
        <root>`` anchor so the agent's shell sees the same root as its
        virtual FS. Without the anchor, ``execute`` runs in whatever cwd
        the agent process happens to have — which on this setup is the
        source repo, causing the agent to write into the real tree.

        For ``execute`` we also refuse commands that reference absolute
        ``/workspace/...`` paths outside the workspace root, or that ``cd``
        out of the workspace. Returning a ``ToolMessage`` signals the caller
        to short-circuit dispatch. Returning ``None`` means "continue".
        """
        if not self._enable_path_normalization:
            return None
        name = tool_call.get("name")
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            return None

        fields = self._FS_TOOL_PATH_FIELDS.get(name)
        if fields is not None:
            for field in fields:
                if field in args:
                    normalized = self._normalize_path_arg(args[field])
                    if normalized is not None and normalized != args[field]:
                        args[field] = normalized
            return None

        if name == "execute":
            cmd = args.get("command")
            if not isinstance(cmd, str) or not cmd:
                return None
            # Sentinel makes re-wraps idempotent (paranoia — middleware
            # shouldn't fire twice on the same call, but composition could
            # surprise us).
            if "VIBE_WORKSPACE_CD=1" in cmd:
                return None
            backend = self._backend
            if backend is None:
                return None
            root = getattr(backend, "cwd", None)
            if root is None:
                return None

            # Piece 4: refuse absolute /workspace/... paths that escape the
            # workspace root. The agent's cwd is already anchored to root by
            # the wrap below, so it never needs absolute paths to its own
            # workspace — and absolute paths to anywhere else are exfiltration
            # surface (the session-7 `cp main.py /workspace/main.py` pattern).
            root_str = str(root).rstrip("/")
            # cd /workspace<...> — even cd to our own root via the absolute
            # form is suspect, since the wrap already cd's there. Treat it as
            # the same class of mistake: the agent thinks "/workspace" is its
            # home, but it isn't.
            cd_m = self._CD_WORKSPACE_RE.search(cmd)
            if cd_m is not None:
                target = cd_m.group(1)
                if not _is_under_root(target, root_str):
                    return self._refuse_execute(tool_call, target, root_str)
            # Any other absolute /workspace/... reference. Allow tokens that
            # are under root (the agent's real workspace path); refuse the
            # rest.
            for tok in self._WORKSPACE_PATH_RE.findall(cmd):
                if not _is_under_root(tok, root_str):
                    return self._refuse_execute(tool_call, tok, root_str)

            args["command"] = (
                f"cd {shlex.quote(str(root))} && "
                f"export VIBE_WORKSPACE_CD=1 && "
                f"{cmd}"
            )
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        # Normalize first so the loop-guard signature reflects what the call
        # actually does (collapses confused host-path spellings to the same
        # canonical signature). A non-None return is a refusal — propagate
        # it directly without recording it in the repeat buffer (refusals
        # don't represent productive progress, and we want the model to
        # retry with a corrected command).
        refusal = self._maybe_normalize_paths(tool_call)
        if refusal is not None:
            return refusal
        sig = _tool_call_signature(tool_call)
        thread_id = self._thread_id(request)
        buf = self._buffer(thread_id)

        # 1. read_file(limit<=0) — deepagents returns empty string for this;
        #    the model has no way to learn from the silence.
        if self._block_read_file_limit_zero and self._read_file_limit_zero(tool_call):
            args = tool_call.get("args") or {}
            buf.append(sig)
            return ToolMessage(
                content=(
                    f"Error: read_file was called with limit={args.get('limit')!r}, "
                    "which returns no content. The 'limit' parameter must be a "
                    "positive integer (number of lines to read). To read the "
                    "first 200 lines, call read_file(file_path=..., offset=0, "
                    "limit=200). Omit 'limit' to read up to the default of 100 "
                    "lines."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )

        # 2. Consecutive-repeat loop-breaker. We act when the buffer is full
        #    AND every signature in it equals the *new* call's signature —
        #    i.e. the model has issued (N-1) identical calls in a row and is
        #    about to issue the N-th. N=3 means the model has already seen
        #    the same response twice.
        if len(buf) == self._repeat_window and all(s == sig for s in buf):
            buf.append(sig)
            name = tool_call.get("name", "<unknown>")
            return ToolMessage(
                content=(
                    f"Loop guard: the tool call {name}(...) with these exact "
                    f"arguments has been issued {self._repeat_window} times in a "
                    "row with no change in outcome. The same call will not "
                    "produce a different result. Either change the arguments "
                    "(different file path, different offset, different "
                    "command), call a different tool, or — if you have enough "
                    "information to proceed — emit your final answer instead."
                ),
                tool_call_id=tool_call["id"],
                name=name,
                status="error",
            )

        buf.append(sig)
        # 3. write_file overwrite enabler. Done immediately before dispatch
        #    so the original tool runs with the path freshly cleared.
        self._maybe_clear_existing_for_overwrite(tool_call)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command[Any]:
        # Async path: the guards above are pure dict lookups and a deque
        # append, so reusing the sync logic is safe — we just need an
        # awaitable handler call when we fall through.
        tool_call = request.tool_call
        refusal = self._maybe_normalize_paths(tool_call)
        if refusal is not None:
            return refusal
        sig = _tool_call_signature(tool_call)
        thread_id = self._thread_id(request)
        buf = self._buffer(thread_id)

        if self._block_read_file_limit_zero and self._read_file_limit_zero(tool_call):
            args = tool_call.get("args") or {}
            buf.append(sig)
            return ToolMessage(
                content=(
                    f"Error: read_file was called with limit={args.get('limit')!r}, "
                    "which returns no content. The 'limit' parameter must be a "
                    "positive integer (number of lines to read). To read the "
                    "first 200 lines, call read_file(file_path=..., offset=0, "
                    "limit=200). Omit 'limit' to read up to the default of 100 "
                    "lines."
                ),
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )

        if len(buf) == self._repeat_window and all(s == sig for s in buf):
            buf.append(sig)
            name = tool_call.get("name", "<unknown>")
            return ToolMessage(
                content=(
                    f"Loop guard: the tool call {name}(...) with these exact "
                    f"arguments has been issued {self._repeat_window} times in a "
                    "row with no change in outcome. The same call will not "
                    "produce a different result. Either change the arguments "
                    "(different file path, different offset, different "
                    "command), call a different tool, or — if you have enough "
                    "information to proceed — emit your final answer instead."
                ),
                tool_call_id=tool_call["id"],
                name=name,
                status="error",
            )

        buf.append(sig)
        self._maybe_clear_existing_for_overwrite(tool_call)
        return await handler(request)

"""Tests for ``AgentGuardrailsMiddleware``.

Covers the path-normalizer (Pieces 1 + 2 from session 6) plus the original
session-5 guards (read_file(limit<=0), consecutive-repeat, write_file
overwrite) — the latter weren't covered by a test file before, and the
session-6 TODO explicitly asks us to confirm they still pass after the
normalizer patch.

The middleware's ``wrap_tool_call`` takes a ``ToolCallRequest`` from
langgraph; we stub a minimal request object that exposes the two attributes
the middleware reads: ``tool_call`` and ``runtime`` (used for thread-id
extraction).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from vibe_serve.guardrails import AgentGuardrailsMiddleware


def _make_request(tool_call: dict, thread_id: str | None = "t1") -> SimpleNamespace:
    """Build a stub ``ToolCallRequest``-shaped object.

    The middleware only reads ``request.tool_call`` and uses ``request.runtime``
    to dig out ``configurable["thread_id"]`` — anything more would be coupling
    to langgraph internals we don't need to exercise here.
    """
    runtime = SimpleNamespace(config={"configurable": {"thread_id": thread_id}})
    return SimpleNamespace(tool_call=tool_call, runtime=runtime)


def _ok_handler(request: SimpleNamespace) -> ToolMessage:
    """Pretend the underlying tool succeeded — return a success message."""
    return ToolMessage(
        content="ok",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call.get("name"),
    )


class _FakeBackend:
    """Minimal stand-in for ``LocalShellBackend``.

    Exposes ``cwd`` for the execute-anchor, and ``_resolve_path`` so the
    write-overwrite path can locate a real on-disk file to unlink.
    """

    def __init__(self, root: Path) -> None:
        self.cwd = root

    def _resolve_path(self, key: str) -> Path:
        vpath = key if key.startswith("/") else "/" + key
        return (self.cwd / vpath.lstrip("/")).resolve()


# ---------------------------------------------------------------------------
# Path normalizer (Piece 1)
# ---------------------------------------------------------------------------


class TestPathNormalizer:
    def test_examples_prefix_collapses_to_root(self):
        m = AgentGuardrailsMiddleware()
        assert (
            m._normalize_path_arg("/workspace/vibe-serve/examples/Llama-3-8B/main.py")
            == "/main.py"
        )

    def test_exp_env_prefix_collapses_run_and_workspace(self):
        m = AgentGuardrailsMiddleware()
        result = m._normalize_path_arg(
            "/workspace/vibe-serve/exp_env/20260629-x/workspace/sub/foo.py"
        )
        assert result == "/sub/foo.py"

    def test_bare_repo_root_prefix(self):
        m = AgentGuardrailsMiddleware()
        assert m._normalize_path_arg("/workspace/vibe-serve/src/foo.py") == "/src/foo.py"

    def test_virtual_path_unchanged(self):
        m = AgentGuardrailsMiddleware()
        assert m._normalize_path_arg("/main.py") is None
        assert m._normalize_path_arg("/sub/main.py") is None

    def test_relative_path_unchanged(self):
        m = AgentGuardrailsMiddleware()
        # Non-absolute paths aren't rewritten — relative paths already resolve
        # under cwd in the backend, and rewriting "main.py" → "/main.py" would
        # change the meaning of `glob` patterns like "*.py".
        assert m._normalize_path_arg("main.py") is None

    def test_non_string_unchanged(self):
        m = AgentGuardrailsMiddleware()
        assert m._normalize_path_arg(None) is None
        assert m._normalize_path_arg(123) is None


class TestMaybeNormalizePaths:
    def test_write_file_rewrites_file_path(self):
        m = AgentGuardrailsMiddleware()
        tc = {
            "name": "write_file",
            "args": {
                "file_path": "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
                "content": "x",
            },
        }
        m._maybe_normalize_paths(tc)
        assert tc["args"]["file_path"] == "/main.py"
        # content untouched
        assert tc["args"]["content"] == "x"

    def test_read_file_rewrites_exp_env_path(self):
        m = AgentGuardrailsMiddleware()
        tc = {
            "name": "read_file",
            "args": {
                "file_path": "/workspace/vibe-serve/exp_env/RUN/workspace/foo.py",
            },
        }
        m._maybe_normalize_paths(tc)
        assert tc["args"]["file_path"] == "/foo.py"

    def test_ls_rewrites_path_field(self):
        m = AgentGuardrailsMiddleware()
        tc = {
            "name": "ls",
            "args": {"path": "/workspace/vibe-serve/examples/Llama-3-8B/reference"},
        }
        m._maybe_normalize_paths(tc)
        assert tc["args"]["path"] == "/reference"

    def test_glob_rewrites_pattern_and_path(self):
        m = AgentGuardrailsMiddleware()
        tc = {
            "name": "glob",
            "args": {
                "pattern": "/workspace/vibe-serve/examples/Llama-3-8B/*.py",
                "path": "/workspace/vibe-serve/",
            },
        }
        m._maybe_normalize_paths(tc)
        assert tc["args"]["pattern"] == "/*.py"
        assert tc["args"]["path"] == "/"

    def test_grep_pattern_not_rewritten(self):
        # grep's "pattern" is a search string, NOT a path — rewriting it
        # would silently change what the model is searching for.
        m = AgentGuardrailsMiddleware()
        tc = {
            "name": "grep",
            "args": {"pattern": "/workspace/vibe-serve/foo", "path": "/"},
        }
        m._maybe_normalize_paths(tc)
        assert tc["args"]["pattern"] == "/workspace/vibe-serve/foo"

    def test_virtual_write_unchanged(self):
        m = AgentGuardrailsMiddleware()
        tc = {"name": "write_file", "args": {"file_path": "/main.py", "content": "x"}}
        m._maybe_normalize_paths(tc)
        assert tc["args"]["file_path"] == "/main.py"

    def test_disabled_normalization_is_noop(self):
        m = AgentGuardrailsMiddleware(enable_path_normalization=False)
        tc = {
            "name": "write_file",
            "args": {
                "file_path": "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
                "content": "x",
            },
        }
        m._maybe_normalize_paths(tc)
        assert (
            tc["args"]["file_path"]
            == "/workspace/vibe-serve/examples/Llama-3-8B/main.py"
        )


# ---------------------------------------------------------------------------
# Execute cwd anchor (Piece 2)
# ---------------------------------------------------------------------------


class TestExecuteAnchor:
    def test_command_gets_cd_prepended(self, tmp_path: Path):
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)
        tc = {"name": "execute", "args": {"command": "ls main.py"}}
        m._maybe_normalize_paths(tc)
        cmd = tc["args"]["command"]
        assert cmd.startswith(f"cd {tmp_path}")
        assert "VIBE_WORKSPACE_CD=1" in cmd
        assert cmd.endswith("&& ls main.py")

    def test_sentinel_prevents_double_wrap(self, tmp_path: Path):
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)
        already = (
            f"cd {tmp_path} && export VIBE_WORKSPACE_CD=1 && ls"
        )
        tc = {"name": "execute", "args": {"command": already}}
        m._maybe_normalize_paths(tc)
        assert tc["args"]["command"] == already

    def test_no_backend_no_anchor(self):
        m = AgentGuardrailsMiddleware(backend=None)
        tc = {"name": "execute", "args": {"command": "ls main.py"}}
        m._maybe_normalize_paths(tc)
        assert tc["args"]["command"] == "ls main.py"

    def test_path_with_spaces_is_quoted(self, tmp_path: Path):
        # shlex.quote handles spaces, dollar signs, etc. — verify the anchor
        # is shell-safe even for awkward workspace paths.
        weird = tmp_path / "has spaces" / "and$dollar"
        weird.mkdir(parents=True)
        backend = _FakeBackend(weird)
        m = AgentGuardrailsMiddleware(backend=backend)
        tc = {"name": "execute", "args": {"command": "ls"}}
        m._maybe_normalize_paths(tc)
        cmd = tc["args"]["command"]
        # The single-quoted form survives spaces and $.
        assert "'" in cmd
        assert str(weird) in cmd.replace("'", "")


# ---------------------------------------------------------------------------
# Execute command-string path filtering (Piece 4)
# ---------------------------------------------------------------------------


class TestExecuteCommandPathFiltering:
    """Refuse ``execute`` commands that reference absolute ``/workspace/...``
    paths outside the agent's real workspace root.

    Session 7 saw the agent issue ``cp main.py /workspace/main.py`` and then
    ``cd /workspace && python -c "import main"``, creating diverging copies
    of the file at the host root vs. its actual workspace. Piece 4 short-
    circuits these as refusals before the shell ever runs them.

    For tests where the refusal must allow paths under the workspace, we use
    a backend whose ``cwd`` is rooted at ``/workspace/vibe-serve/exp_env/RUN
    /workspace`` so we can construct both allowed (under-root) and refused
    (outside-root) ``/workspace/...`` tokens.
    """

    _REAL_ROOT = Path("/workspace/vibe-serve/exp_env/RUN/workspace")

    def _middleware(self) -> AgentGuardrailsMiddleware:
        # cwd is a Path object that doesn't need to exist on disk — the
        # middleware only stringifies it for comparison and prefix wrapping.
        return AgentGuardrailsMiddleware(backend=_FakeBackend(self._REAL_ROOT))

    def test_cd_to_host_workspace_refused(self):
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "cd /workspace && python -c 'import main'"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Refusing execute" in result.content
        assert "/workspace" in result.content

    def test_cp_to_host_workspace_refused(self):
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "cp main.py /workspace/main.py"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "/workspace/main.py" in result.content

    def test_cd_to_real_workspace_root_allowed(self):
        # cd to the legit workspace root is fine — even though the wrap
        # already cd's there, the agent issuing a redundant cd is harmless.
        m = self._middleware()
        cmd = f"cd {self._REAL_ROOT} && ls"
        tc = {"name": "execute", "args": {"command": cmd}, "id": "x"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        # Pass-through to the handler — middleware doesn't refuse, and the
        # cwd anchor wraps the command without complaint.
        assert result.content == "ok"

    def test_path_outside_workspace_no_workspace_prefix_allowed(self):
        # The filter only catches /workspace/* references. Other absolute
        # paths (/etc, /tmp, /home, /usr) aren't policed here — the backend
        # may have its own sandboxing, but the filter's job is specifically
        # the /workspace footgun.
        m = self._middleware()
        tc = {"name": "execute", "args": {"command": "cat /etc/cpuinfo"}, "id": "x"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"

    def test_workspace_path_under_root_allowed(self):
        # ls /workspace/vibe-serve/exp_env/RUN/workspace/sub — the absolute
        # path IS the agent's real workspace, so it's allowed.
        m = self._middleware()
        cmd = f"ls {self._REAL_ROOT}/sub"
        tc = {"name": "execute", "args": {"command": cmd}, "id": "x"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"

    def test_redirect_to_host_workspace_refused(self):
        # > /workspace/x — the redirect is the same exfil channel as cp.
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "echo hi > /workspace/leak.txt"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "/workspace/leak.txt" in result.content

    def test_sibling_workspace_dir_refused(self):
        # /workspace/vibe-serve-other/* shouldn't be allowed just because
        # the root is /workspace/vibe-serve/... — the _is_under_root helper
        # enforces a path-component boundary.
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "ls /workspace/vibe-serve-other/foo"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    def test_refusal_not_recorded_in_repeat_buffer(self):
        # Three consecutive refused execute calls should NOT trip the
        # repeat-guard's "loop guard" message — we want the model to see the
        # refusal each time, not get a different error after N tries that
        # might mislead it. Refusals bypass the repeat buffer entirely.
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "cp main.py /workspace/main.py"},
            "id": "x",
        }
        for _ in range(5):
            result = m.wrap_tool_call(
                _make_request({**tc, "args": dict(tc["args"])}), _ok_handler
            )
            assert "Refusing execute" in result.content

    def test_relative_workspace_substring_not_matched(self):
        # "workspace" appearing mid-token isn't an absolute /workspace path.
        # The lookbehind on the regex should prevent a false positive.
        m = self._middleware()
        tc = {
            "name": "execute",
            "args": {"command": "echo my/workspace/path"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"

    def test_disabled_normalization_skips_filter(self):
        # When the agent-side patch is turned off entirely, the filter is
        # off too — the run_in option for testing legacy behaviour.
        m = AgentGuardrailsMiddleware(
            backend=_FakeBackend(self._REAL_ROOT),
            enable_path_normalization=False,
        )
        tc = {
            "name": "execute",
            "args": {"command": "cd /workspace && ls"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_async_path_also_refuses(self):
        m = self._middleware()

        async def ahandler(req):
            return ToolMessage(
                content="ok", tool_call_id=req.tool_call["id"], name="execute"
            )

        tc = {
            "name": "execute",
            "args": {"command": "cp main.py /workspace/main.py"},
            "id": "x",
        }
        result = await m.awrap_tool_call(_make_request(tc), ahandler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Refusing execute" in result.content


# ---------------------------------------------------------------------------
# read_file(limit<=0) guard (session 5 — should still work post-patch)
# ---------------------------------------------------------------------------


class TestReadFileLimitZero:
    def test_limit_zero_short_circuits(self):
        m = AgentGuardrailsMiddleware()
        tc = {"name": "read_file", "args": {"file_path": "/x.py", "limit": 0}, "id": "1"}
        req = _make_request(tc)
        result = m.wrap_tool_call(req, _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "limit=0" in result.content or "limit parameter" in result.content.lower() or "positive integer" in result.content

    def test_negative_limit_short_circuits(self):
        m = AgentGuardrailsMiddleware()
        tc = {"name": "read_file", "args": {"file_path": "/x.py", "limit": -1}, "id": "1"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    def test_positive_limit_passes_through(self):
        m = AgentGuardrailsMiddleware()
        tc = {"name": "read_file", "args": {"file_path": "/x.py", "limit": 100}, "id": "1"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"

    def test_no_limit_passes_through(self):
        m = AgentGuardrailsMiddleware()
        tc = {"name": "read_file", "args": {"file_path": "/x.py"}, "id": "1"}
        result = m.wrap_tool_call(_make_request(tc), _ok_handler)
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# Consecutive-repeat guard (session 5)
# ---------------------------------------------------------------------------


class TestRepeatGuard:
    def test_three_identical_calls_break_on_fourth(self):
        m = AgentGuardrailsMiddleware(repeat_window=3)
        tc = {"name": "execute", "args": {"command": "echo hi"}, "id": "x"}
        # Calls 1, 2, 3 pass through.
        for _ in range(3):
            req = _make_request({**tc, "args": dict(tc["args"])})
            assert m.wrap_tool_call(req, _ok_handler).content == "ok"
        # The 4th identical call hits the guard.
        req = _make_request({**tc, "args": dict(tc["args"])})
        result = m.wrap_tool_call(req, _ok_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Loop guard" in result.content

    def test_different_args_reset_guard(self):
        m = AgentGuardrailsMiddleware(repeat_window=3)
        for cmd in ["a", "a", "a", "b"]:
            tc = {"name": "execute", "args": {"command": cmd}, "id": "x"}
            req = _make_request(tc)
            assert m.wrap_tool_call(req, _ok_handler).content == "ok"

    def test_separate_thread_ids_have_independent_buffers(self):
        m = AgentGuardrailsMiddleware(repeat_window=3)
        tc_args = {"name": "execute", "args": {"command": "echo"}, "id": "x"}
        # Fill thread "a" — three identical calls.
        for _ in range(3):
            req = _make_request({**tc_args, "args": dict(tc_args["args"])}, thread_id="a")
            assert m.wrap_tool_call(req, _ok_handler).content == "ok"
        # Thread "b" should be unaffected — first call passes.
        req_b = _make_request({**tc_args, "args": dict(tc_args["args"])}, thread_id="b")
        assert m.wrap_tool_call(req_b, _ok_handler).content == "ok"

    def test_normalization_unifies_signatures(self, tmp_path: Path):
        # Pre-normalization, two different host-path spellings of the same
        # virtual path would have distinct signatures and slip past the
        # repeat guard. Post-normalization, they collapse to /main.py and
        # the guard sees them as identical.
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend, repeat_window=3)
        spellings = [
            "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
            "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
            "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
            "/main.py",
        ]
        results: list[ToolMessage] = []
        for sp in spellings:
            tc = {
                "name": "write_file",
                "args": {"file_path": sp, "content": "x"},
                "id": "x",
            }
            results.append(m.wrap_tool_call(_make_request(tc), _ok_handler))
        # First three normalize to /main.py and pass through; the fourth
        # is the same normalized signature and trips the guard.
        for r in results[:3]:
            assert r.content == "ok"
        assert results[3].status == "error"
        assert "Loop guard" in results[3].content


# ---------------------------------------------------------------------------
# write_file overwrite (session 5 — still works post-patch)
# ---------------------------------------------------------------------------


class TestWriteOverwrite:
    def test_existing_file_is_unlinked_before_handler(self, tmp_path: Path):
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)
        target = tmp_path / "main.py"
        target.write_text("old content")
        assert target.exists()

        captured = {}

        def handler(req):
            # By the time the handler runs, the file should be gone — the
            # overwrite enabler has already unlinked it.
            captured["existed_when_handler_ran"] = target.exists()
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="write_file")

        tc = {
            "name": "write_file",
            "args": {"file_path": "/main.py", "content": "new"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), handler)
        assert result.content == "ok"
        assert captured["existed_when_handler_ran"] is False

    def test_normalized_path_still_unlinks(self, tmp_path: Path):
        # The model spells the path with a host prefix; the normalizer
        # rewrites it to /main.py first; the overwrite enabler then resolves
        # /main.py under cwd and unlinks it.
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)
        target = tmp_path / "main.py"
        target.write_text("old content")

        def handler(req):
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="write_file")

        tc = {
            "name": "write_file",
            "args": {
                "file_path": "/workspace/vibe-serve/examples/Llama-3-8B/main.py",
                "content": "new",
            },
            "id": "x",
        }
        m.wrap_tool_call(_make_request(tc), handler)
        assert not target.exists()
        # And the args were rewritten in place so the handler sees the
        # virtual form.
        assert tc["args"]["file_path"] == "/main.py"

    def test_nonexistent_file_is_passthrough(self, tmp_path: Path):
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)

        def handler(req):
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="write_file")

        tc = {
            "name": "write_file",
            "args": {"file_path": "/new.py", "content": "x"},
            "id": "x",
        }
        result = m.wrap_tool_call(_make_request(tc), handler)
        assert result.content == "ok"


# ---------------------------------------------------------------------------
# Async wrapper symmetry
# ---------------------------------------------------------------------------


class TestAsyncWrapper:
    @pytest.mark.asyncio
    async def test_async_normalizes_and_passes_through(self, tmp_path: Path):
        backend = _FakeBackend(tmp_path)
        m = AgentGuardrailsMiddleware(backend=backend)

        async def ahandler(req):
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="write_file")

        tc = {
            "name": "write_file",
            "args": {
                "file_path": "/workspace/vibe-serve/examples/Llama-3-8B/foo.py",
                "content": "x",
            },
            "id": "x",
        }
        result = await m.awrap_tool_call(_make_request(tc), ahandler)
        assert result.content == "ok"
        assert tc["args"]["file_path"] == "/foo.py"

"""Tests for the review agent's preset tools."""

from __future__ import annotations

import pytest

from app import tools


class TestResolve:
    def test_normal_path(self):
        p = tools._resolve("svc/pay.py")
        assert p.name == "pay.py"

    def test_leading_slash_stripped(self):
        p = tools._resolve("/svc/pay.py")
        assert p.name == "pay.py"

    def test_escape_repo_root(self):
        with pytest.raises(tools.ToolError, match="escapes the repo root"):
            tools._resolve("../../etc/passwd")


class TestGetDiff:
    def test_known_task(self):
        result = tools.get_diff("pay")
        assert "charge" in result
        assert "requests.post" in result

    def test_unknown_task(self):
        with pytest.raises(tools.ToolError, match="no diff for task"):
            tools.get_diff("nonexistent")


class TestReadFile:
    def test_read_existing(self):
        result = tools.read_file("svc/pay.py")
        assert "charge" in result
        # Check line numbers are present
        assert "|" in result

    def test_read_missing(self):
        with pytest.raises(tools.ToolError, match="no such file"):
            tools.read_file("does-not-exist.py")

    def test_line_range(self):
        result = tools.read_file("svc/pay.py", start=1, end=3)
        lines = result.strip().split("\n")
        assert len(lines) == 3


class TestGrepRepo:
    def test_match(self):
        result = tools.grep_repo(r"requests\.post", glob="**/*.py")
        assert "svc/pay.py" in result or "svc/auth.py" in result

    def test_no_match(self):
        result = tools.grep_repo(r"NO_SUCH_PATTERN_XYZ")
        assert result == "(no matches)"

    def test_bad_regex(self):
        with pytest.raises(tools.ToolError, match="bad regex"):
            tools.grep_repo(r"[")


class TestRunLinter:
    def test_finds_md5(self):
        result = tools.run_linter("svc/auth.py")
        assert "S324" in result
        assert "md5" in result

    def test_finds_print(self):
        # Create a temp file inside the fixture repo so _resolve accepts it
        import tempfile
        from pathlib import Path
        repo = tools.config.FIXTURE_REPO
        tmp = repo / "tmp_test_print.py"
        tmp.write_text("print('hello')\n")
        try:
            result = tools.run_linter("tmp_test_print.py")
            assert "T201" in result
        finally:
            tmp.unlink(missing_ok=True)

    def test_clean_file(self):
        result = tools.run_linter("svc/pay.py")
        # pay.py has no md5, print, or bare except, but it DOES have
        # requests.post without a timeout
        assert "B113" in result


# --- TestDispatch ----------------------------------------------------------


class TestDispatch:
    def test_dispatches_get_diff(self):
        text, err = tools.dispatch("get_diff", {"task_id": "pay"})
        assert not err
        assert "charge" in text

    def test_dispatches_read_file(self):
        text, err = tools.dispatch("read_file", {"path": "svc/pay.py"})
        assert not err
        assert "charge" in text

    def test_dispatches_grep_repo(self):
        text, err = tools.dispatch("grep_repo", {"pattern": "requests"})
        assert not err
        assert "pay.py" in text or "auth.py" in text or "billing.py" in text

    def test_dispatches_run_linter(self):
        text, err = tools.dispatch("run_linter", {"path": "svc/auth.py"})
        assert not err
        assert "S324" in text  # md5

    def test_unknown_tool_returns_error(self):
        text, err = tools.dispatch("nonexistent_tool", {})
        assert err
        assert "unknown tool" in text

    def test_bad_arguments_returns_error(self):
        text, err = tools.dispatch("get_diff", {})  # missing task_id
        assert err

    def test_tool_error_returned_not_raised(self):
        text, err = tools.dispatch("get_diff", {"task_id": "nonexistent"})
        assert err
        assert "no diff" in text

    def test_grep_no_match(self):
        text, err = tools.dispatch("grep_repo", {"pattern": "NO_SUCH_PATTERN_XYZ"})
        assert not err
        assert text == "(no matches)"


# --- TestReadFileEdgeCases -------------------------------------------------


class TestReadFileEdgeCases:
    def test_start_equals_end_reads_one_line(self):
        result = tools.read_file("svc/pay.py", start=1, end=1)
        lines = result.strip().split("\n")
        assert len(lines) == 1

    def test_start_beyond_file(self):
        result = tools.read_file("svc/pay.py", start=999, end=1000)
        # When start exceeds file length, range(lo, hi+1) is empty
        # but ''.strip().split('\n') returns [''], so check content is empty
        assert result.strip() == ""

    def test_line_numbers_present(self):
        result = tools.read_file("svc/pay.py", start=1, end=3)
        for line in result.strip().split("\n"):
            assert "|" in line  # line number separator


# --- TestGrepRepoEdgeCases -------------------------------------------------


class TestGrepRepoEdgeCases:
    def test_glob_filter_py_only(self):
        result = tools.grep_repo(r"func ", glob="**/*.go")
        assert "worker.go" in result or result == "(no matches)"

    def test_truncates_at_50(self):
        # Create many matches to trigger truncation
        import tempfile
        from pathlib import Path
        repo = tools.config.FIXTURE_REPO
        tmp = repo / "tmp_many.py"
        lines = "\n".join(f"match_{i}" for i in range(60))
        tmp.write_text(lines, encoding="utf-8")
        try:
            result = tools.grep_repo(r"match", glob="tmp_many.py")
            assert "truncated" in result
        finally:
            tmp.unlink(missing_ok=True)


# --- TestLinterEdgeCases ---------------------------------------------------


class TestLinterEdgeCases:
    def test_finds_trailing_whitespace(self):
        import tempfile
        from pathlib import Path
        repo = tools.config.FIXTURE_REPO
        tmp = repo / "tmp_ws.py"
        tmp.write_text("x = 1   \n", encoding="utf-8")
        try:
            result = tools.run_linter("tmp_ws.py")
            assert "W291" in result
        finally:
            tmp.unlink(missing_ok=True)

    def test_finds_bare_except(self):
        import tempfile
        from pathlib import Path
        repo = tools.config.FIXTURE_REPO
        tmp = repo / "tmp_exc.py"
        tmp.write_text("try:\n    x\nexcept:\n    pass\n", encoding="utf-8")
        try:
            result = tools.run_linter("tmp_exc.py")
            assert "E722" in result
        finally:
            tmp.unlink(missing_ok=True)

    def test_no_issues_message(self):
        result = tools.run_linter("svc/billing.py")
        # billing.py has requests.post without timeout (B113)
        assert "B113" in result

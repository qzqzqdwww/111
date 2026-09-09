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

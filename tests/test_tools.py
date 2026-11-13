"""Safety boundaries of the tools and project copying."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from langchain_core.tools import ToolException

from agent.config import Settings
from agent.runner import _files_note, _ignore_for_copy, _mode, _write_patch
from agent.tools import PageBudget, _resolve, file_tools, sandbox_args


def tools_by_name(workspace):
    return {t.name: t for t in file_tools(workspace)}


@pytest.mark.parametrize("path", ["../outside.txt", "a/../../outside.txt", "C:/Windows/evil.txt", ".git/config", "sub/.git/hooks/x"])
def test_paths_outside_the_workspace_are_refused(tmp_path, path):
    with pytest.raises(ToolException):
        _resolve(tmp_path, path)


def test_symlink_cannot_escape_the_workspace(tmp_path):
    outside = tmp_path / "outside"
    workspace = tmp_path / "workspace"
    outside.mkdir()
    workspace.mkdir()
    try:
        os.symlink(outside, workspace / "link", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks needs extra permissions on this machine")
    with pytest.raises(ToolException):
        _resolve(workspace, "link/secret.txt")


def test_page_budget_limits_sections_per_page():
    budget = PageBudget(max_sections=2)
    budget.take("https://a.org/doc")
    budget.take("https://a.org/doc/#part")
    with pytest.raises(ToolException, match="limit"):
        budget.take("https://a.org/doc")
    budget.take("https://b.org/other")


def test_write_read_and_edit(tmp_path):
    tools = tools_by_name(tmp_path)
    tools["write_file"].invoke({"path": "sub/a.txt", "content": "hello world"})
    assert tools["read_file"].invoke({"path": "sub/a.txt"}) == "hello world"
    tools["edit_file"].invoke({"path": "sub/a.txt", "old_text": "world", "new_text": "there"})
    assert (tmp_path / "sub/a.txt").read_text() == "hello there"
    assert "sub/a.txt" in tools["list_files"].invoke({})


def test_tool_errors_are_returned_to_the_model(tmp_path):
    result = tools_by_name(tmp_path)["read_file"].invoke({"path": "../x"})
    assert "outside the workspace" in result


def test_handing_in_a_file_returns_all_of_it(tmp_path):
    finish = tools_by_name(tmp_path)["finish_with_file"]
    (tmp_path / "long.md").write_bytes(b"line\n" * 3000)
    assert finish.invoke({"path": "long.md"}) == "line\n" * 3000, "not cut to one section"
    (tmp_path / "empty.md").write_text("  \n")
    assert "empty" in finish.invoke({"path": "empty.md"})
    assert "No such file" in finish.invoke({"path": "missing.md"})
    assert "outside the workspace" in finish.invoke({"path": "../x.md"})
    (tmp_path / "huge.md").write_text("x" * 50_000)
    assert "too long" in finish.invoke({"path": "huge.md"})


def test_report_points_to_files_left_in_the_workspace(tmp_path):
    assert _files_note(tmp_path) == ""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/a.txt").write_text("x")
    assert "- sub/a.txt" in _files_note(tmp_path)


def test_runs_saved_before_modes_existed_still_resume_correctly():
    assert _mode({"moltbook": True}) == "moltbook"
    assert _mode({"use_web": True}) == "agent"
    assert _mode({"mode": "research"}) == "research"


def test_edit_requires_a_unique_match(tmp_path):
    (tmp_path / "a.txt").write_text("x x")
    result = tools_by_name(tmp_path)["edit_file"].invoke({"path": "a.txt", "old_text": "x", "new_text": "y"})
    assert "exactly once" in result
    assert (tmp_path / "a.txt").read_text() == "x x"


def test_sandbox_has_no_network_or_privileges(tmp_path):
    args = sandbox_args(Settings(), tmp_path, "name", "ls")
    joined = " ".join(args)
    for flag in ["--network none", "--cap-drop ALL", "--read-only", "no-new-privileges", "--user 1000:1000"]:
        assert flag in joined


def test_secrets_and_git_are_not_copied(tmp_path):
    (tmp_path / "real").write_text("x")
    names = [".env", ".env.local", ".git", "node_modules", "main.py", ".env.example", ".gitignore"]
    ignored = _ignore_for_copy(str(tmp_path), names)
    assert {".env", ".env.local", ".env.example", ".git", "node_modules"} <= ignored
    assert "main.py" not in ignored
    assert ".gitignore" not in ignored


def test_edit_keeps_windows_line_endings(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"one\r\ntwo\r\nthree\r\n")
    tools_by_name(tmp_path)["edit_file"].invoke({"path": "a.txt", "old_text": "one\ntwo", "new_text": "1\n2"})
    assert (tmp_path / "a.txt").read_bytes() == b"1\r\n2\r\nthree\r\n"


def test_write_keeps_the_line_endings_given(tmp_path):
    tools_by_name(tmp_path)["write_file"].invoke({"path": "a.txt", "content": "one\ntwo\n"})
    assert (tmp_path / "a.txt").read_bytes() == b"one\ntwo\n"


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_patch_applies_to_the_original_project(tmp_path, line_ending, autocrlf):
    project = tmp_path / "project"
    run_dir = tmp_path / "run"
    project.mkdir()
    (project / "a.txt").write_bytes(b"old" + line_ending)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "config", "core.autocrlf", autocrlf], cwd=project, check=True)
    shutil.copytree(project, run_dir / "baseline", ignore=shutil.ignore_patterns(".git"))
    shutil.copytree(project, run_dir / "workspace", ignore=shutil.ignore_patterns(".git"))
    (run_dir / "workspace/a.txt").write_bytes(b"new" + line_ending)
    (run_dir / "workspace/added.txt").write_bytes(b"added\n")

    _write_patch(run_dir)

    subprocess.run(["git", "apply", "-p2", str(run_dir / "changes.patch")], cwd=project, check=True)
    # With autocrlf on, git itself may choose Windows line endings for the result.
    assert (project / "a.txt").read_bytes().rstrip() == b"new"
    assert (project / "added.txt").exists()

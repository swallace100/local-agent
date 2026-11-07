"""Runs real containers to confirm the sandbox's limits. Skipped if Docker isn't running."""

import subprocess
from dataclasses import replace

import pytest

from agent.config import Settings
from agent.tools import command_tool

docker_running = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not docker_running, reason="Docker isn't running")


async def run(tmp_path, command, timeout=60):
    settings = replace(Settings(), command_timeout_seconds=timeout)
    return await command_tool(settings, tmp_path).ainvoke({"command": command})


async def test_workspace_is_writable_and_shared(tmp_path):
    output = await run(tmp_path, "echo hi > note.txt")
    assert output.startswith("Exit code: 0")
    assert (tmp_path / "note.txt").read_text().strip() == "hi"


async def test_no_network(tmp_path):
    code = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"
    output = await run(tmp_path, f'python -c "{code}"')
    assert not output.startswith("Exit code: 0")


async def test_system_is_read_only(tmp_path):
    output = await run(tmp_path, "touch /usr/x")
    assert "Read-only file system" in output


async def test_runs_as_non_root(tmp_path):
    assert "\n1000" in await run(tmp_path, "id -u")


async def test_timeout_returns_a_message_and_removes_the_container(tmp_path):
    output = await run(tmp_path, "sleep 60", timeout=5)
    assert "timed out" in output
    leftovers = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=agent-sandbox"], capture_output=True, text=True
    )
    assert leftovers.stdout.strip() == ""

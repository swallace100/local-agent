"""The agent's tools. Nothing here can touch your machine outside the run's workspace.

Files are limited to the workspace, and commands run in a Docker container that
has no network and can only see the workspace.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.tools import BaseTool, StructuredTool, ToolException, tool

from agent.config import ROOT, Settings
from agent.rag import Knowledge

MAX_OUTPUT_CHARS = 8_000
# A file handed in as the result is read by the reviewer in one piece, so it must fit.
MAX_RESULT_CHARS = 40_000
WEB_SERVER = ROOT / "tools" / "web-search" / "server.mjs"


def _resolve(workspace: Path, relative: str) -> Path:
    """Return a path inside the workspace, refusing anything that escapes it."""
    target = (workspace / relative).resolve()
    if not target.is_relative_to(workspace.resolve()):
        raise ToolException(f"Path is outside the workspace: {relative}")
    if ".git" in target.relative_to(workspace.resolve()).parts:
        raise ToolException("The .git folder can't be accessed.")
    return target


def _cap(text: str, offset: int = 0) -> str:
    chunk = text[offset : offset + MAX_OUTPUT_CHARS]
    end = offset + len(chunk)
    if end < len(text):
        chunk += f"\n\n[Showing {offset}-{end} of {len(text)} characters. Use offset={end} for more.]"
    return chunk


def _read(path: Path) -> str:
    # newline="" keeps line endings as they are, so edits don't rewrite the whole file.
    with path.open(encoding="utf-8", errors="replace", newline="") as file:
        return file.read()


def _write(path: Path, content: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)


def file_tools(workspace: Path) -> list[BaseTool]:
    @tool
    def list_files() -> str:
        """List the files in the workspace."""
        root = workspace.resolve()
        paths = [
            p.relative_to(root).as_posix()
            for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts
        ]
        return _cap("\n".join(paths)) or "The workspace is empty."

    @tool
    def read_file(path: str, offset: int = 0) -> str:
        """Read a text file from the workspace."""
        target = _resolve(workspace, path)
        if not target.is_file():
            raise ToolException(f"No such file: {path}")
        return _cap(_read(target), offset)

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace."""
        target = _resolve(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write(target, content)
        return f"Wrote {len(content)} characters to {path}"

    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace text in a workspace file. old_text must match exactly and appear once."""
        target = _resolve(workspace, path)
        if not target.is_file():
            raise ToolException(f"No such file: {path}")
        content = _read(target)
        if "\r\n" in content:
            # The model writes \n, so match the file's Windows line endings.
            old_text = old_text.replace("\r\n", "\n").replace("\n", "\r\n")
            new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
        count = content.count(old_text)
        if count != 1:
            raise ToolException(f"old_text must appear exactly once in {path}, found {count}.")
        _write(target, content.replace(old_text, new_text))
        return f"Edited {path}"

    @tool
    def finish_with_file(path: str) -> str:
        """End the task with a workspace file as your final result. Use this instead of a
        reply only when the result is a document you already saved with write_file. The user
        reads the whole file. For a short answer, just reply."""
        target = _resolve(workspace, path)
        if not target.is_file():
            raise ToolException(f"No such file: {path}")
        content = _read(target)
        if not content.strip():
            raise ToolException(f"{path} is empty. Write the result to it first.")
        if len(content) > MAX_RESULT_CHARS:
            raise ToolException(
                f"{path} is too long to hand in ({len(content)} characters, and the limit is "
                f"{MAX_RESULT_CHARS}). Reply with a summary that names the file instead."
            )
        return content

    all_tools = [list_files, read_file, write_file, edit_file, finish_with_file]
    for t in all_tools:
        t.handle_tool_error = True
    return all_tools


def sandbox_args(settings: Settings, workspace: Path, name: str, command: str) -> list[str]:
    """Build the docker command: no network, no privileges, only the workspace is writable."""
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--memory", "2g",
        "--cpus", "2",
        "--pids-limit", "256",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs", "/tmp",
        "--user", "1000:1000",
        "--env", "HOME=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--volume", f"{workspace.resolve()}:/workspace",
        "--workdir", "/workspace",
        settings.sandbox_image,
        "sh", "-c", command,
    ]  # fmt: skip


def command_tool(settings: Settings, workspace: Path) -> BaseTool:
    @tool
    async def run_command(command: str) -> str:
        """Run a shell command in a Linux sandbox with Python, such as: python -m unittest.
        The sandbox's working directory is the workspace, and it has no internet access."""
        name = f"agent-sandbox-{uuid.uuid4().hex[:12]}"
        process = await asyncio.create_subprocess_exec(
            *sandbox_args(settings, workspace, name, command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(
                process.communicate(), timeout=settings.command_timeout_seconds
            )
        except TimeoutError:
            # Killing the docker client wouldn't stop the container itself.
            remover = await asyncio.create_subprocess_exec(
                "docker", "rm", "-f", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )  # fmt: skip
            await remover.wait()
            try:
                process.kill()
            except ProcessLookupError:
                pass  # Removing the container already ended the docker client.
            await process.wait()
            return f"Command timed out after {settings.command_timeout_seconds} seconds."
        text = output.decode("utf-8", errors="replace")
        return f"Exit code: {process.returncode}\n{_cap(text)}"

    return run_command


def source_label(document) -> str:
    """How a knowledge chunk is cited: its file, and its page when the file has pages."""
    page = document.metadata.get("page")
    return f"knowledge/{document.metadata['source']}" + (f", page {page}" if page else "")


def knowledge_tool(knowledge: Knowledge) -> BaseTool:
    @tool
    async def knowledge_search(query: str) -> str:
        """Search the user's private knowledge folder. Try this first for questions that
        could be about the user's own projects, notes or documents."""
        results = await asyncio.to_thread(knowledge.search, query)
        if not results:
            return "Nothing relevant in the knowledge folder."
        return "\n\n".join(f"Source: {source_label(doc)}\n{doc.page_content}" for doc, _ in results)

    return knowledge_search


class PageBudget:
    """Limits how many sections of one page the agent reads, so a long page can't use up a run."""

    def __init__(self, max_sections: int):
        self.max_sections = max_sections
        self.sections_read: dict[str, int] = {}

    def take(self, url: str) -> None:
        key = url.split("#")[0].rstrip("/")
        used = self.sections_read.get(key, 0)
        if used >= self.max_sections:
            raise ToolException(
                f"You've already read {used} sections of this page, which is the limit. "
                "Use what you have, or read a different source."
            )
        self.sections_read[key] = used + 1


@asynccontextmanager
async def web_tools(settings: Settings) -> AsyncIterator[list[BaseTool]]:
    """Connect to the same web server OpenCode uses, so there's one implementation."""
    transport = StdioTransport(
        command="node",
        args=[str(WEB_SERVER)],
        env={"SEARXNG_URL": settings.searxng_url},
    )
    budget = PageBudget(settings.max_sections_per_page)
    async with Client(transport) as client:

        async def call(name: str, arguments: dict) -> str:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
            if result.is_error:
                raise ToolException(text)
            return text

        async def web_search(query: str, max_results: int = 5) -> str:
            """Search the web. Returns titles, URLs and short snippets. Snippets are not
            enough to cite: read promising pages with web_read_page."""
            return await call("search", {"query": query, "max_results": max_results})

        async def web_read_page(url: str, offset: int = 0) -> str:
            """Read a public web page as clean text. If the result says there's more,
            call again with the given offset only if you need it."""
            budget.take(url)
            return await call("read_page", {"url": url, "offset": offset})

        yield [
            StructuredTool.from_function(coroutine=web_search, handle_tool_error=True),
            StructuredTool.from_function(coroutine=web_read_page, handle_tool_error=True),
        ]

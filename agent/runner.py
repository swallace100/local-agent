"""Starts and resumes runs, and saves their results to runs/<id>/."""

import asyncio
import json
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.config import Settings
from agent.graph import WORK_PROMPT, build_graph, initial_state, strip_narration
from agent.memory import Memory
from agent.moltbook import WORK_PROMPT as MOLTBOOK_WORK_PROMPT
from agent.moltbook import add_plug_section, moltbook_tools
from agent.organize import build_organize_graph, initial_organize_state, save_moves
from agent.rag import Knowledge
from agent.remote import BoardGone, BoardServer, RemoteBoard
from agent.research import Researcher, build_research_graph, initial_research_state
from agent.team import _work, team_stage
from agent.tools import command_tool, file_tools, knowledge_tool, web_tools

# Never copied into a run, so the agent can't read secrets or a repo's git config.
MAX_FILES_LISTED = 20
# Both follow the research workflow. A team run has several agents do the gathering.
RESEARCH_MODES = ("research", "team")
PROJECT_IGNORE = {".git", ".venv", "venv", "node_modules", "__pycache__", ".data", "runs"}


@dataclass
class RunResult:
    run_id: str
    status: str
    run_dir: Path


def preflight(settings: Settings, need_web: bool, need_docker: bool = True) -> None:
    """Fail immediately with a useful message, rather than partway through an unattended run."""
    checks = [(f"{settings.ollama_url}/api/version", "Ollama isn't running. Start the Ollama app.")]
    if need_web:
        checks.append((settings.searxng_url, "SearXNG isn't running. Run: docker compose up -d"))
    for url, message in checks:
        try:
            urllib.request.urlopen(url, timeout=5)
        except OSError as error:
            raise SystemExit(f"{message} ({error})") from error
    if need_docker and subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("Docker isn't running, and commands need it. Start Docker Desktop.")


def make_model(settings: Settings) -> ChatOllama:
    return ChatOllama(model=settings.model, base_url=settings.ollama_url, temperature=0.2, reasoning=False)


def _ignore_for_copy(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        is_env_file = name == ".env" or name.startswith(".env.")
        if name in PROJECT_IGNORE or is_env_file or Path(directory, name).is_symlink():
            ignored.add(name)
    return ignored


def _new_run(
    settings: Settings,
    task: str,
    project: Path | None,
    use_web: bool,
    mode: str,
    folder: Path | None = None,
    host: dict | None = None,
) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run_dir = settings.runs_dir / run_id
    workspace = run_dir / "workspace"
    if project:
        shutil.copytree(project, workspace, ignore=_ignore_for_copy)
        shutil.copytree(workspace, run_dir / "baseline")
    else:
        workspace.mkdir(parents=True)
    meta = {
        "task": task,
        "project": str(project) if project else None,
        "use_web": use_web,
        "mode": mode,
        "folder": str(folder) if folder else None,
        "host": host,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def _write_patch(run_dir: Path) -> None:
    """Save the agent's changes as a patch, with your project left untouched.

    Compares two plain folders outside any repo. The sandbox could have written to the
    workspace, so git must never treat it as a repo (a planted hook would run on this machine).
    """
    result = subprocess.run(
        # autocrlf off, or git would change line endings and the patch wouldn't apply.
        ["git", "-c", "core.autocrlf=false", "diff", "--no-index", "--no-color", "--binary",
         "baseline", "workspace"],
        cwd=run_dir, capture_output=True,
    )  # fmt: skip
    if result.stdout:
        (run_dir / "changes.patch").write_bytes(result.stdout)


def _save_results(run_dir: Path, state: dict, seconds: float) -> None:
    warnings = state.get("warnings") or []
    report = strip_narration(state.get("report", ""))
    if warnings:
        notes = "\n".join(f"- {w}" for w in warnings).replace("\n", "\n> ")
        report = f"> **Warnings from the agent's checks:**\n>\n> {notes}\n\n{report}"
    if (run_dir / "baseline").exists():
        _write_patch(run_dir)
    else:
        report += _files_note(run_dir / "workspace")
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    summary = {
        "status": state.get("status"),
        "steps": state.get("steps"),
        "revisions": state.get("revisions"),
        "seconds": round(seconds),
        "warnings": warnings,
        "plan": state.get("plan"),
    }
    (run_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _files_note(workspace: Path) -> str:
    """Point to files the agent left in its workspace, which the report alone wouldn't show."""
    if not workspace.is_dir():
        return ""
    files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file())
    if not files:
        return ""
    listed = "\n".join(f"- {name}" for name in files[:MAX_FILES_LISTED])
    if len(files) > MAX_FILES_LISTED:
        listed += f"\n- and {len(files) - MAX_FILES_LISTED} more"
    return f"\n\n---\n\nFiles the agent wrote, in `{workspace}`:\n\n{listed}\n"


def _mode(meta: dict) -> str:
    # Runs made before modes existed only recorded whether they were Moltbook runs.
    return meta.get("mode") or ("moltbook" if meta.get("moltbook") else "agent")


def _log(trace_file, node: str, update: dict | None) -> None:
    """Save every step for later debugging, and print a one-line summary."""
    update = update or {}
    entries = []
    for message in update.get("messages", []):
        entries.append(
            {
                "type": message.type,
                "name": getattr(message, "name", None),
                "tool_calls": getattr(message, "tool_calls", None),
                "content": str(message.content),
            }
        )
    record = {"time": time.time(), "node": node, "messages": entries}
    if update.get("trace"):
        record["trace"] = update["trace"]
    trace_file.write(json.dumps(record) + "\n")
    trace_file.flush()
    calls = [
        f"{c['name']}({json.dumps(c['args'])[:90]})"
        for e in entries
        for c in (e["tool_calls"] or [])
    ]
    detail = ", ".join(calls) or update.get("activity") or ""
    print(f"[{datetime.now():%H:%M:%S}] {node}" + (f": {detail}" if detail else ""), flush=True)


class _open_to_joiners:
    """Called by a team once its tasks are posted: serves the board and waits for joiners."""

    def __init__(self, settings: Settings, run_dir: Path, host: dict, tools: list):
        self.settings, self.run_dir, self.host = settings, run_dir, host
        read_tool = next(tool for tool in tools if tool.name == "web_read_page")
        self.read_page = read_tool.metadata["read_unmetered"]
        self.server: BoardServer | None = None

    async def __call__(self, questions: list[str]) -> None:
        if self.server:  # The step is being retried, and the board is already being served.
            return
        self.server = BoardServer(
            self.run_dir / "board.sqlite",
            questions,
            self.read_page,
            asyncio.get_running_loop(),
            lease_seconds=self.settings.team_lease_minutes * 60,
            max_attempts=self.settings.team_attempts,
            listen=self.host["listen"],
            port=self.host["port"],
        )
        address = f"http://{_this_machine()}:{self.server.port}"
        print(f"\nOther machines can join this run with:\n  agent join {address} --key {self.server.key}\n", flush=True)
        waiting_for, deadline = self.host.get("wait_for", 0), time.time() + self.host.get("wait_seconds", 0)
        while len(self.server.joined) < waiting_for and time.time() < deadline:
            await asyncio.sleep(1)
        if waiting_for:
            print(f"{len(self.server.joined)} joined. Starting.", flush=True)


def _this_machine() -> str:
    """This machine's address on the local network, for the joining instructions."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))  # Nothing is sent. It only picks the outgoing interface.
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


async def join_team(settings: Settings, url: str, key: str, name: str) -> None:
    """Work on another machine's team run, using this machine's model and web access."""
    preflight(settings, need_web=True, need_docker=False)
    board = RemoteBoard(url, key, name)
    try:
        questions = board.questions()
        print(f"Joined as {board.name}. {len(questions)} questions on the board.", flush=True)
        started = time.time()
        async with web_tools(settings) as web:
            researcher = Researcher(make_model(settings), web, settings)

            def over() -> bool:
                return (time.time() - started) / 60 >= settings.max_minutes or researcher.steps >= settings.team_max_steps

            await _work(board.name, board, researcher, questions, over, settings)
        print("Nothing left to do. Leaving.")
    except BoardGone as gone:
        print(f"Stopped: {gone}")
    finally:
        board.close()


async def _execute(settings: Settings, run_dir: Path, graph_input, *, knowledge=None) -> RunResult:
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    mode = _mode(meta)
    preflight(settings, need_web=meta["use_web"], need_docker=mode == "agent")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / "workspace"
    config = {"configurable": {"thread_id": run_dir.name}, "recursion_limit": 500}

    if mode == "moltbook":
        # Bot-written content must never share a run with notes, files or commands.
        tools, work_prompt = moltbook_tools(), MOLTBOOK_WORK_PROMPT
    elif mode == "organize":
        # The model only answers questions about a list of files. It gets no tools at all.
        tools, work_prompt = [], ""
    elif mode in RESEARCH_MODES:
        # Research reads and reports. It has no files to change and no commands to run.
        tools, work_prompt = [knowledge_tool(knowledge or Knowledge(settings))], ""
    else:
        knowledge = knowledge or Knowledge(settings)
        tools = [*file_tools(workspace), command_tool(settings, workspace), knowledge_tool(knowledge)]
        work_prompt = WORK_PROMPT
    model = make_model(settings)
    # Web research remembers what it found, and starts from what earlier runs found.
    memory = Memory(settings) if mode in RESEARCH_MODES and meta["use_web"] and settings.remember else None

    started = time.time()
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:

        async def run_graph(all_tools):
            if mode == "organize":
                graph = build_organize_graph(model, settings, checkpointer=saver)
            elif mode in RESEARCH_MODES:
                # A team shares one board, kept with the run so a resumed run carries on from it.
                hosting = _open_to_joiners(settings, run_dir, meta["host"], all_tools) if meta.get("host") else None
                servers.append(hosting)
                team = team_stage(run_dir / "board.sqlite", settings, hosting) if mode == "team" else None
                leads = partial(memory.leads, limit=settings.pages_per_question) if memory else None
                graph = build_research_graph(
                    model, all_tools, settings, checkpointer=saver, team=team, leads=leads
                )
            else:
                graph = build_graph(
                    model,
                    all_tools,
                    settings,
                    checkpointer=saver,
                    work_prompt=work_prompt,
                    files_are_the_result=mode == "agent" and not meta.get("project"),
                )
            if graph_input is None:
                # Restart the time budget, since time spent stopped shouldn't count.
                await graph.aupdate_state(config, {"started_at": time.time()})
            with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as trace:
                async for event in graph.astream(graph_input, config, stream_mode="updates"):
                    for node, update in event.items():
                        _log(trace, node, update)
            return (await graph.aget_state(config)).values

        servers: list = []
        try:
            if meta["use_web"]:
                async with web_tools(settings) as web:
                    state = await run_graph([*tools, *web])
            else:
                state = await run_graph(tools)
        finally:
            for hosting in servers:
                if hosting and hosting.server:
                    hosting.server.close()

    if mode == "moltbook":
        state["report"] = add_plug_section(strip_narration(state.get("report", "")), state["messages"])
    if mode == "organize":
        save_moves(run_dir, state)
    if memory and state.get("notes"):
        saved = memory.remember(run_dir.name, state["task"], state["questions"], state["notes"])
        print(f"Remembered {saved} notes for later runs.", flush=True)
    _save_results(run_dir, state, time.time() - started)
    return RunResult(run_dir.name, state.get("status") or "unknown", run_dir)


async def start_run(
    settings: Settings,
    task: str,
    project: Path | None = None,
    use_web: bool = True,
    knowledge: Knowledge | None = None,
    mode: str = "agent",
    folder: Path | None = None,
    on_start: Callable[[str], None] | None = None,
    host: dict | None = None,
) -> RunResult:
    """Start a run. mode is "agent" (tools and a sandbox), "research", "team", "organize" or "moltbook".

    folder is the real folder an organize run plans for. The run reads it but never changes it.
    on_start is told the run's ID as soon as it exists, so an interrupted run can be found again.
    host, for a team run, opens the board to agents on other machines (see agent/remote.py).
    """
    run_dir = _new_run(settings, task, project, use_web, mode, folder, host)
    print(f"Run {run_dir.name}", flush=True)
    if on_start:
        on_start(run_dir.name)
    if mode == "organize":
        state = initial_organize_state(task, folder, settings)
    elif mode in RESEARCH_MODES:
        state = initial_research_state(task)
    else:
        state = initial_state(task)
    return await _execute(settings, run_dir, state, knowledge=knowledge)


async def resume_run(settings: Settings, run_id: str) -> RunResult:
    run_dir = settings.runs_dir / run_id
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"No such run: {run_id}")
    return await _execute(settings, run_dir, None)

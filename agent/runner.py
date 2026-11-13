"""Starts and resumes runs, and saves their results to runs/<id>/."""

import json
import shutil
import subprocess
import time
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.config import Settings
from agent.graph import WORK_PROMPT, build_graph, initial_state, strip_narration
from agent.moltbook import WORK_PROMPT as MOLTBOOK_WORK_PROMPT
from agent.moltbook import add_plug_section, moltbook_tools
from agent.rag import Knowledge
from agent.research import build_research_graph, initial_research_state
from agent.tools import command_tool, file_tools, knowledge_tool, web_tools

# Never copied into a run, so the agent can't read secrets or a repo's git config.
MAX_FILES_LISTED = 20
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


def _ignore_for_copy(directory: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        is_env_file = name == ".env" or name.startswith(".env.")
        if name in PROJECT_IGNORE or is_env_file or Path(directory, name).is_symlink():
            ignored.add(name)
    return ignored


def _new_run(settings: Settings, task: str, project: Path | None, use_web: bool, mode: str) -> Path:
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
    elif mode == "research":
        # Research reads and reports. It has no files to change and no commands to run.
        tools, work_prompt = [knowledge_tool(knowledge or Knowledge(settings))], ""
    else:
        knowledge = knowledge or Knowledge(settings)
        tools = [*file_tools(workspace), command_tool(settings, workspace), knowledge_tool(knowledge)]
        work_prompt = WORK_PROMPT
    model = ChatOllama(
        model=settings.model, base_url=settings.ollama_url, temperature=0.2, reasoning=False
    )

    started = time.time()
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db)) as saver:

        async def run_graph(all_tools):
            if mode == "research":
                graph = build_research_graph(model, all_tools, settings, checkpointer=saver)
            else:
                graph = build_graph(model, all_tools, settings, checkpointer=saver, work_prompt=work_prompt)
            if graph_input is None:
                # Restart the time budget, since time spent stopped shouldn't count.
                await graph.aupdate_state(config, {"started_at": time.time()})
            with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as trace:
                async for event in graph.astream(graph_input, config, stream_mode="updates"):
                    for node, update in event.items():
                        _log(trace, node, update)
            return (await graph.aget_state(config)).values

        if meta["use_web"]:
            async with web_tools(settings) as web:
                state = await run_graph([*tools, *web])
        else:
            state = await run_graph(tools)

    if mode == "moltbook":
        state["report"] = add_plug_section(strip_narration(state.get("report", "")), state["messages"])
    _save_results(run_dir, state, time.time() - started)
    return RunResult(run_dir.name, state.get("status") or "unknown", run_dir)


async def start_run(
    settings: Settings,
    task: str,
    project: Path | None = None,
    use_web: bool = True,
    knowledge: Knowledge | None = None,
    mode: str = "agent",
) -> RunResult:
    """Start a run. mode is "agent" (tools and a sandbox), "research" or "moltbook"."""
    run_dir = _new_run(settings, task, project, use_web, mode)
    print(f"Run {run_dir.name}", flush=True)
    state = initial_research_state(task) if mode == "research" else initial_state(task)
    return await _execute(settings, run_dir, state, knowledge=knowledge)


async def resume_run(settings: Settings, run_id: str) -> RunResult:
    run_dir = settings.runs_dir / run_id
    if not (run_dir / "run.json").is_file():
        raise SystemExit(f"No such run: {run_id}")
    return await _execute(settings, run_dir, None)

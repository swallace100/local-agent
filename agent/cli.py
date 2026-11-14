"""Command line: agent run | research | team | organize | apply | undo | resume | runs | index | search | eval | moltbook"""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from agent.config import Settings
from agent.moltbook import digest_task
from agent.organize import DEFAULT_INSTRUCTION, MOVES_FILE, apply_moves, scan, undo_moves, why_not
from agent.rag import Knowledge
from agent.runner import resume_run, start_run
from agent.tools import source_label


def _print_result(result) -> None:
    print(f"\nStatus: {result.status}")
    print(f"Report: {result.run_dir / 'report.md'}")
    if (result.run_dir / MOVES_FILE).exists():
        print(f"Nothing has been moved. Read the plan, then apply it with: agent apply {result.run_id}")
    patch = result.run_dir / "changes.patch"
    if patch.exists():
        print(f"Code changes: {patch}")
        print(f"Apply them from your project folder with: git apply -p2 \"{patch}\"")


def _list_runs(settings: Settings) -> None:
    if not settings.runs_dir.exists():
        print("No runs yet.")
        return
    for run_dir in sorted(settings.runs_dir.iterdir()):
        meta_file, result_file = run_dir / "run.json", run_dir / "result.json"
        if not meta_file.is_file():
            continue
        task = json.loads(meta_file.read_text(encoding="utf-8"))["task"]
        status = "unfinished"
        if result_file.is_file():
            status = json.loads(result_file.read_text(encoding="utf-8"))["status"]
        print(f"{run_dir.name}  {status:<20} {task[:70]}")


def _print_moves(settings: Settings, run_id: str, action, done: str) -> None:
    run_dir = settings.runs_dir / run_id
    if not run_dir.is_dir():
        raise SystemExit(f"No such run: {run_id}")
    result = action(run_dir)
    print(f"{done} {result.moved} files.")
    if result.skipped:
        print(f"Skipped {len(result.skipped)}:")
        for reason in result.skipped:
            print(f"  {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Unattended local agent")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a task")
    run.add_argument("task")
    run.add_argument("--project", type=Path, help="Work on a copy of this folder and produce a patch")
    run.add_argument("--no-web", action="store_true", help="Turn off web search and page reading")
    run.add_argument("--max-minutes", type=int, help="Time limit for this run")

    research = commands.add_parser(
        "research", help="Research a question step by step and write a cited report"
    )
    research.add_argument("task")
    research.add_argument("--no-web", action="store_true", help="Use only your notes in knowledge/")
    research.add_argument("--max-minutes", type=int, help="Time limit for this run")

    team = commands.add_parser(
        "team", help="Research a question with several agents that share notes and cross-check each other"
    )
    team.add_argument("task")
    team.add_argument("--agents", type=int, help="How many agents work on it (default 2)")
    team.add_argument("--no-web", action="store_true", help="Use only your notes in knowledge/")
    team.add_argument("--max-minutes", type=int, help="Time limit for this run")

    organize = commands.add_parser(
        "organize", help="Plan how to tidy a folder into subfolders, without moving anything"
    )
    organize.add_argument("folder", type=Path)
    organize.add_argument("instruction", nargs="?", default=DEFAULT_INSTRUCTION)
    organize.add_argument("--max-minutes", type=int, help="Time limit for this run")

    apply = commands.add_parser("apply", help="Move the files as an organize run planned")
    apply.add_argument("run_id")

    undo = commands.add_parser("undo", help="Put back the files that apply moved")
    undo.add_argument("run_id")

    resume = commands.add_parser("resume", help="Continue an interrupted run")
    resume.add_argument("run_id")

    commands.add_parser("runs", help="List runs")
    commands.add_parser("index", help="Update the search index from the knowledge folder")

    search = commands.add_parser("search", help="Search the knowledge folder directly")
    search.add_argument("query")

    evaluate = commands.add_parser("eval", help="Score the agent on the tasks in evals/tasks.json")
    evaluate.add_argument("--only", help="Run a single eval by ID")
    evaluate.add_argument("--no-web", action="store_true", help="Skip the tasks that search the web")

    moltbook = commands.add_parser("moltbook", help="Write a read-only digest of Moltbook")
    moltbook.add_argument("--topic", help="Focus the digest on one subject")

    args = parser.parse_args()
    settings = Settings()

    if args.command in ("run", "research", "team", "organize") and args.max_minutes:
        settings = replace(settings, max_minutes=args.max_minutes)

    if args.command == "run":
        if args.project and not args.project.is_dir():
            raise SystemExit(f"No such folder: {args.project}")
        _print_result(asyncio.run(start_run(settings, args.task, args.project, not args.no_web)))
    elif args.command == "research":
        _print_result(asyncio.run(start_run(settings, args.task, use_web=not args.no_web, mode="research")))
    elif args.command == "team":
        if args.agents is not None:
            if args.agents < 2:
                raise SystemExit("A team needs at least 2 agents. For one agent, use: agent research")
            settings = replace(settings, team_agents=args.agents)
        _print_result(asyncio.run(start_run(settings, args.task, use_web=not args.no_web, mode="team")))
    elif args.command == "organize":
        folder = args.folder.resolve()
        if not folder.is_dir():
            raise SystemExit(f"No such folder: {args.folder}")
        if why_not(folder):
            raise SystemExit(why_not(folder))
        if not scan(folder, 1)[0]:
            raise SystemExit(f"There are no loose files in {folder} to organize.")
        _print_result(
            asyncio.run(start_run(settings, args.instruction, use_web=False, mode="organize", folder=folder))
        )
    elif args.command == "apply":
        _print_moves(settings, args.run_id, apply_moves, "Moved")
        print(f"To reverse this, run: agent undo {args.run_id}")
    elif args.command == "undo":
        _print_moves(settings, args.run_id, undo_moves, "Put back")
    elif args.command == "moltbook":
        _print_result(asyncio.run(start_run(settings, digest_task(args.topic), use_web=False, mode="moltbook")))
    elif args.command == "resume":
        _print_result(asyncio.run(resume_run(settings, args.run_id)))
    elif args.command == "runs":
        _list_runs(settings)
    elif args.command == "index":
        result = Knowledge(settings).index()
        print(
            f"Indexed {result.added} new or changed files, removed {result.removed}, "
            f"{result.unchanged} unchanged. Total chunks: {result.chunks}"
        )
        for reason in result.skipped:
            print(f"Skipped {reason}")
    elif args.command == "search":
        for doc, distance in Knowledge(settings).search(args.query):
            # A chunk with no distance was found by its words alone.
            match = "words" if distance is None else f"{distance:.3f}"
            print(f"\n[{match}] {source_label(doc)}\n{doc.page_content}")
    elif args.command == "eval":
        from agent.evals import run_evals

        raise SystemExit(asyncio.run(run_evals(settings, args.only, skip_web=args.no_web)))


if __name__ == "__main__":
    main()

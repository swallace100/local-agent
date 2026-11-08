"""Command line: agent run | resume | runs | index | search | eval | moltbook"""

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from agent.config import Settings
from agent.moltbook import digest_task
from agent.rag import Knowledge
from agent.runner import resume_run, start_run


def _print_result(result) -> None:
    print(f"\nStatus: {result.status}")
    print(f"Report: {result.run_dir / 'report.md'}")
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Unattended local agent")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a task")
    run.add_argument("task")
    run.add_argument("--project", type=Path, help="Work on a copy of this folder and produce a patch")
    run.add_argument("--no-web", action="store_true", help="Turn off web search and page reading")
    run.add_argument("--max-minutes", type=int, help="Time limit for this run")

    resume = commands.add_parser("resume", help="Continue an interrupted run")
    resume.add_argument("run_id")

    commands.add_parser("runs", help="List runs")
    commands.add_parser("index", help="Update the search index from the knowledge folder")

    search = commands.add_parser("search", help="Search the knowledge folder directly")
    search.add_argument("query")

    evaluate = commands.add_parser("eval", help="Score the agent on the tasks in evals/tasks.json")
    evaluate.add_argument("--only", help="Run a single eval by ID")

    moltbook = commands.add_parser("moltbook", help="Write a read-only digest of Moltbook")
    moltbook.add_argument("--topic", help="Focus the digest on one subject")

    args = parser.parse_args()
    settings = Settings()

    if args.command == "run":
        if args.max_minutes:
            settings = replace(settings, max_minutes=args.max_minutes)
        if args.project and not args.project.is_dir():
            raise SystemExit(f"No such folder: {args.project}")
        _print_result(asyncio.run(start_run(settings, args.task, args.project, not args.no_web)))
    elif args.command == "moltbook":
        _print_result(asyncio.run(start_run(settings, digest_task(args.topic), use_web=False, moltbook=True)))
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
    elif args.command == "search":
        for doc, distance in Knowledge(settings).search(args.query):
            print(f"\n[{distance:.3f}] {doc.metadata['source']}\n{doc.page_content}")
    elif args.command == "eval":
        from agent.evals import run_evals

        raise SystemExit(asyncio.run(run_evals(settings, args.only)))


if __name__ == "__main__":
    main()

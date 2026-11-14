"""Command line: agent run | research | team | organize | apply | undo | queue | recall | resume | runs | index | search
| eval | moltbook"""

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from agent.config import Settings
from agent.jobs import MODES, Queue, run_queue, schedule, scheduled_time, unschedule
from agent.memory import Memory
from agent.moltbook import digest_task
from agent.organize import DEFAULT_INSTRUCTION, MOVES_FILE, apply_moves, scan, undo_moves, why_not
from agent.rag import Knowledge
from agent import moltbook_commands
from agent.moltbook_account import Account, load_key
from agent.moltbook_auto import (
    MAX_COMMENTS,
    auto_state,
    check_in,
    checkins_scheduled,
    schedule_checkins,
    unschedule_checkins,
)
from agent.runner import make_model, preflight, resume_run, start_run
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


class _Tee:
    """Copies everything printed into a log file, for runs nobody is watching."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def fileno(self) -> int:
        # Programs the agent starts, such as the web tools, write their errors straight to a
        # file handle. They get the log, which is the last stream.
        return self.streams[-1].fileno()

    def isatty(self) -> bool:
        return False


def _also_log_to(path: Path | None, what: str) -> None:
    """Copy everything printed from here on into a file, for runs started by a schedule."""
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("a", encoding="utf-8")
    log.write(f"\n===== {what} started {datetime.now():%Y-%m-%d %H:%M} =====\n")
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, log) if sys.__stdout__ else log


def _moltbook_auto(settings: Settings, args) -> None:
    state = auto_state(settings)
    if args.action == "auto" and args.switch == "on":
        if not load_key():
            raise SystemExit('The agent hasn\'t joined Moltbook yet. Run: agent moltbook join "Name" "Description"')
        state.on, state.failed_puzzles, state.stopped_because = True, 0, ""
        state.save()
        print("On. The agent may now comment on Moltbook without asking you first.")
        if args.every:
            schedule_checkins(settings, args.every)
            print(f"Windows will run a check-in every {args.every} hours. Each one is written up in runs/.")
        else:
            print("Run a check-in with: agent moltbook checkin    (or add --every 6 to have Windows do it)")
    elif args.action == "auto" and args.switch == "off":
        state.on = False
        state.save()
        unschedule_checkins()
        print("Off. The agent will not post anything unless you approve it, and no check-ins are scheduled.")
    elif args.action == "auto":
        print("On." if state.on else "Off." + (f" It was turned off because {state.stopped_because}" if state.stopped_because else ""))
        print(f"Scheduled check-ins: {'yes' if checkins_scheduled() else 'no'}")
    else:
        saved = load_key()
        if not saved:
            raise SystemExit('The agent hasn\'t joined Moltbook yet. Run: agent moltbook join "Name" "Description"')
        _also_log_to(args.log, "Check-in")
        preflight(settings, need_web=False, need_docker=False)
        report = asyncio.run(
            check_in(settings, make_model(settings), Account(saved[0]), saved[1], args.dry_run, args.max)
        )
        print(f"\nWritten up in {report}")


def _moltbook_command(settings: Settings, args) -> None:
    if args.action in ("auto", "checkin"):
        return _moltbook_auto(settings, args)
    if args.action == "join":
        moltbook_commands.join(args.name, args.description)
    elif args.action == "status":
        moltbook_commands.status()
    elif args.action == "outbox":
        moltbook_commands.show(settings)
    elif args.action == "reject":
        moltbook_commands.reject(settings, args.id)
    elif args.action == "inbox":
        moltbook_commands.inbox()
    elif args.action == "draft":
        saved = load_key()
        if not saved:
            raise SystemExit('The agent hasn\'t joined Moltbook yet. Run: agent moltbook join "Name" "Description"')
        preflight(settings, need_web=False, need_docker=False)
        asyncio.run(
            moltbook_commands.draft(
                settings, make_model(settings), saved[1], args.post, args.new, args.submolt, args.reply_to
            )
        )
    elif args.action == "approve":
        preflight(settings, need_web=False, need_docker=False)
        asyncio.run(
            moltbook_commands.approve(
                settings, make_model(settings), args.id, args.edit, answer_later=args.answer_later
            )
        )
    elif args.action == "answer":
        moltbook_commands.answer(settings, args.number)


def _queue_command(settings: Settings, args) -> None:
    queue = Queue(settings.queue_file)
    if args.action == "add":
        if args.mode != "moltbook" and not args.task:
            raise SystemExit("Give the task to queue, in quotes.")
        if args.project and not args.project.is_dir():
            raise SystemExit(f"No such folder: {args.project}")
        project = str(args.project.resolve()) if args.project else None
        job = queue.add(args.mode, args.task or "", not args.no_web, project, args.repeat)
        print(f"Queued as task {job.id}. Start the queue with: agent queue run")
    elif args.action == "list":
        when = scheduled_time()
        print(f"The queue runs every day at {when}." if when else "The queue isn't scheduled. Run it with: agent queue run")
        for job in queue.jobs:
            repeat = " (repeats)" if job.repeat else ""
            outcome = f"  last: {job.outcome.splitlines()[0][:60]}" if job.outcome else ""
            print(f"{job.id:>3}  {job.status:<8} {job.mode:<9} {job.task[:60] or 'Moltbook digest'}{repeat}{outcome}")
        if not queue.jobs:
            print("The queue is empty.")
    elif args.action == "remove":
        print("Removed." if queue.remove(args.id) else f"There is no task {args.id}.")
    elif args.action == "clear":
        print(f"Cleared {queue.clear_finished()} finished tasks.")
    elif args.action == "run":
        _also_log_to(args.log, "Queue")
        asyncio.run(run_queue(settings))
    elif args.action == "schedule":
        schedule(settings, args.time)
        print(f"Windows will run the queue every day at {args.time}, waking this PC if it's asleep.")
        print("Ollama must be running then, and Docker Desktop too for tasks that search the web or run commands.")
    elif args.action == "unschedule":
        print("The nightly run is off." if unschedule() else "The queue wasn't scheduled.")


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
    research.add_argument("--no-memory", action="store_true", help="Don't use or add to what earlier runs found")

    team = commands.add_parser(
        "team", help="Research a question with several agents that share notes and cross-check each other"
    )
    team.add_argument("task")
    team.add_argument("--agents", type=int, help="How many agents work on it (default 2)")
    team.add_argument("--no-web", action="store_true", help="Use only your notes in knowledge/")
    team.add_argument("--max-minutes", type=int, help="Time limit for this run")
    team.add_argument("--no-memory", action="store_true", help="Don't use or add to what earlier runs found")

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

    queue = commands.add_parser("queue", help="Line up tasks and work through them unattended")
    actions = queue.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add", help="Add a task to the queue")
    add.add_argument("task", nargs="?", help="The task, or the topic for a Moltbook digest")
    add.add_argument("--mode", choices=MODES, default="research", help="How to run it (default: research)")
    add.add_argument("--no-web", action="store_true", help="Turn off web search and page reading")
    add.add_argument("--project", type=Path, help="With --mode run: work on a copy of this folder")
    add.add_argument("--repeat", action="store_true", help="Run it again every time the queue runs")
    actions.add_parser("list", help="Show the queue")
    remove = actions.add_parser("remove", help="Take a task out of the queue")
    remove.add_argument("id", type=int)
    actions.add_parser("clear", help="Take finished tasks out of the queue")
    run_queued = actions.add_parser("run", help="Work through everything waiting")
    run_queued.add_argument("--log", type=Path, help="Also write the output to this file")
    at = actions.add_parser("schedule", help="Run the queue every day at a set time")
    at.add_argument("time", help="24-hour time, such as 02:00")
    actions.add_parser("unschedule", help="Stop running the queue every day")

    recall = commands.add_parser("recall", help="Search what earlier research runs found")
    recall.add_argument("query")

    resume = commands.add_parser("resume", help="Continue an interrupted run")
    resume.add_argument("run_id")

    commands.add_parser("runs", help="List runs")
    commands.add_parser("index", help="Update the search index from the knowledge folder")

    search = commands.add_parser("search", help="Search the knowledge folder directly")
    search.add_argument("query")

    evaluate = commands.add_parser("eval", help="Score the agent on the tasks in evals/tasks.json")
    evaluate.add_argument("--only", help="Run a single eval by ID")
    evaluate.add_argument("--no-web", action="store_true", help="Skip the tasks that search the web")

    moltbook = commands.add_parser(
        "moltbook", help="Write a digest of Moltbook, or manage what the agent says there"
    )
    moltbook.add_argument("--topic", help="Focus the digest on one subject")
    social = moltbook.add_subparsers(dest="action")
    join = social.add_parser("join", help="Register the agent on Moltbook")
    join.add_argument("name")
    join.add_argument("description")
    social.add_parser("status", help="Check whether you have claimed the agent's account")
    draft = social.add_parser("draft", help="Have the agent draft a comment or a post for you to approve")
    draft.add_argument("post", nargs="?", help="The ID or address of the post to comment on")
    draft.add_argument("--reply-to", help="The ID of a comment to reply to, within that post")
    draft.add_argument("--new", help="Draft a new post instead, about what you describe here")
    draft.add_argument("--submolt", default="general", help="With --new: the community to post in")
    social.add_parser("outbox", help="Show the drafts waiting for you")
    approve = social.add_parser("approve", help="Post a draft")
    approve.add_argument("id", type=int)
    approve.add_argument("--edit", help="Post this text instead of the draft's")
    approve.add_argument(
        "--answer-later",
        action="store_true",
        help="Show Moltbook's puzzle and stop, to answer with: agent moltbook answer",
    )
    answer = social.add_parser("answer", help="Answer the puzzle left by approve --answer-later")
    answer.add_argument("number", type=float)
    reject = social.add_parser("reject", help="Drop a draft")
    reject.add_argument("id", type=int)
    social.add_parser("inbox", help="Show replies to what the agent has posted")
    auto = social.add_parser("auto", help="Turn commenting without your approval on or off")
    auto.add_argument("switch", choices=["on", "off", "status"])
    auto.add_argument("--every", type=int, metavar="HOURS", help="With on: have Windows run a check-in this often")
    checkin = social.add_parser("checkin", help="Let the agent read Moltbook and comment where it has something to say")
    checkin.add_argument("--dry-run", action="store_true", help="Show what it would post, without posting")
    checkin.add_argument("--max", type=int, default=MAX_COMMENTS, help="The most comments to post this time")
    checkin.add_argument("--log", type=Path, help="Also write the output to this file")

    args = parser.parse_args()
    settings = Settings()

    if args.command in ("run", "research", "team", "organize") and args.max_minutes:
        settings = replace(settings, max_minutes=args.max_minutes)

    if getattr(args, "no_memory", False):
        settings = replace(settings, remember=False)

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
    elif args.command == "queue":
        _queue_command(settings, args)
    elif args.command == "recall":
        found = Memory(settings).recall(args.query)
        for doc, _ in found:
            question, fact, quote = doc.page_content.split("\n", 2)
            print(f"\n{fact}\n  {quote}\n  {doc.metadata['source']}")
            print(f"  Found {doc.metadata['date']} while researching: {question}")
        if not found:
            print("Nothing from earlier research matches that.")
    elif args.command == "moltbook" and args.action:
        _moltbook_command(settings, args)
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

"""A queue of tasks to work through unattended, such as overnight.

    agent queue add "question"     add a task
    agent queue run                work through everything waiting, one task at a time
    agent queue schedule 02:00     have Windows run the queue every night

Each task is an ordinary run, so its report lands in runs/ like any other. When the queue
finishes, one summary file lists what was done. A task interrupted by a crash or a reboot
is picked up from its last completed step the next time the queue runs.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from agent.config import ROOT, Settings
from agent.moltbook import digest_task
from agent.runner import resume_run, start_run

# What can be queued. Organizing a folder is left out, because its plan needs reading first.
MODES = ("research", "team", "run", "moltbook")
SCHEDULED_TASK = "local-agent queue"
CLOCK_TIME = re.compile(r"([01]?\d|2[0-3]):[0-5]\d")
FINISHED = ("done", "failed")
TASK_CHARS = 70


@dataclass
class Job:
    id: int
    mode: str
    task: str
    use_web: bool = True
    project: str | None = None
    # A repeating task goes back to waiting after each run, such as a nightly digest.
    repeat: bool = False
    status: str = "waiting"  # waiting, running, done or failed
    run_id: str | None = None
    # How the last run ended, or why it failed.
    outcome: str = ""
    added: str = ""
    finished: str = ""


class Queue:
    """The queued tasks, kept in one JSON file and saved after every change."""

    def __init__(self, path: Path):
        self.path = path
        self.jobs = [Job(**entry) for entry in json.loads(path.read_text(encoding="utf-8"))] if path.is_file() else []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Written to one side and swapped in, so a crash can't leave half a file.
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps([asdict(job) for job in self.jobs], indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def add(self, mode: str, task: str, use_web: bool = True, project: str | None = None, repeat: bool = False) -> Job:
        job = Job(
            id=max((job.id for job in self.jobs), default=0) + 1,
            mode=mode,
            task=task,
            use_web=use_web,
            project=project,
            repeat=repeat,
            added=datetime.now().isoformat(timespec="seconds"),
        )
        self.jobs.append(job)
        self.save()
        return job

    def remove(self, job_id: int) -> bool:
        kept = [job for job in self.jobs if job.id != job_id]
        removed = len(kept) < len(self.jobs)
        self.jobs = kept
        self.save()
        return removed

    def clear_finished(self) -> int:
        kept = [job for job in self.jobs if job.status not in FINISHED]
        cleared = len(self.jobs) - len(kept)
        self.jobs = kept
        self.save()
        return cleared

    def pending(self) -> list[Job]:
        """Tasks still to do, with any that were interrupted first."""
        return [j for j in self.jobs if j.status == "running"] + [j for j in self.jobs if j.status == "waiting"]

    def update(self, job: Job, **changes) -> None:
        for name, value in changes.items():
            setattr(job, name, value)
        self.save()


@contextmanager
def _only_one_runner(lock_path: Path):
    """Stop two windows working through the queue at once. The lock goes when the process ends."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise SystemExit("The queue is already being worked through in another window.") from None
    try:
        yield
    finally:
        handle.close()


async def _run_job(settings: Settings, queue: Queue, job: Job) -> str:
    """Run one task and return its status. Carries on an interrupted run where it stopped."""
    interrupted = job.status == "running" and job.run_id
    if interrupted and (settings.runs_dir / job.run_id / "run.json").is_file():
        print(f"Carrying on with run {job.run_id}, which was interrupted.", flush=True)
        return (await resume_run(settings, job.run_id)).status

    queue.update(job, status="running", run_id=None)
    remember_run = lambda run_id: queue.update(job, run_id=run_id)  # noqa: E731
    if job.mode == "moltbook":
        task, mode, use_web = digest_task(job.task or None), "moltbook", False
    else:
        task, mode, use_web = job.task, "agent" if job.mode == "run" else job.mode, job.use_web
    project = Path(job.project) if job.project else None
    result = await start_run(settings, task, project, use_web, mode=mode, on_start=remember_run)
    return result.status


def _summary(settings: Settings, started: datetime, finished: list[Job]) -> Path:
    lines = [f"# Queue results, {started:%d %B %Y %H:%M}", "", "| Task | Result | Report |", "| --- | --- | --- |"]
    for job in finished:
        task = (job.task or "Moltbook digest").replace("|", "/").replace("\n", " ")
        has_report = job.run_id and (settings.runs_dir / job.run_id / "report.md").is_file()
        report = f"runs/{job.run_id}/report.md" if has_report else ""
        lines.append(f"| {task} | {job.outcome.splitlines()[0][:120] if job.outcome else ''} | {report} |")
    path = settings.runs_dir / f"queue-{started:%Y%m%d-%H%M}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def run_queue(settings: Settings) -> Path | None:
    """Work through every waiting task in turn. One failing doesn't stop the rest."""
    with _only_one_runner(settings.data_dir / "queue.lock"):
        queue = Queue(settings.queue_file)
        pending = queue.pending()
        if not pending:
            print("Nothing is waiting in the queue.")
            return None
        started, finished = datetime.now(), []
        for position, job in enumerate(pending):
            if position and settings.queue_pause_seconds:
                await asyncio.sleep(settings.queue_pause_seconds)
            print(f"\n=== Task {position + 1} of {len(pending)}: {job.task[:TASK_CHARS] or 'Moltbook digest'} ===", flush=True)
            try:
                outcome, failed = await _run_job(settings, queue, job), False
            except (Exception, SystemExit) as error:  # A run ends with SystemExit when a service is down.
                outcome, failed = f"failed: {error}", True
                print(outcome, flush=True)
            status = "waiting" if job.repeat else "failed" if failed else "done"
            queue.update(job, status=status, outcome=outcome, finished=datetime.now().isoformat(timespec="seconds"))
            finished.append(job)
        summary = _summary(settings, started, finished)
        print(f"\nQueue finished. Summary: {summary}")
        return summary


def _powershell(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True
    )


def _quoted(text: object) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def schedule(settings: Settings, at: str, task_name: str = SCHEDULED_TASK) -> None:
    """Have Windows run the queue every day at the given time, waking the PC if it's asleep."""
    if os.name != "nt":
        raise SystemExit("Scheduling is only set up for Windows. Elsewhere, add `agent queue run` to cron.")
    if not CLOCK_TIME.fullmatch(at):
        raise SystemExit("Give the time as hours and minutes on a 24-hour clock, such as 02:00.")
    program = Path(sys.executable).with_name("agent.exe")
    log = settings.data_dir / "queue.log"
    script = (
        f"$action = New-ScheduledTaskAction -Execute {_quoted(program)} "
        f"-Argument {_quoted(f'queue run --log \"{log}\"')} -WorkingDirectory {_quoted(ROOT)}; "
        f"$trigger = New-ScheduledTaskTrigger -Daily -At {_quoted(at)}; "
        "$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun "
        "-ExecutionTimeLimit (New-TimeSpan -Hours 8); "
        f"Register-ScheduledTask -TaskName {_quoted(task_name)} -Action $action -Trigger $trigger "
        "-Settings $settings -Force | Out-Null"
    )
    result = _powershell(script)
    if result.returncode != 0:
        raise SystemExit(f"Windows couldn't schedule the queue: {result.stderr.strip()}")


def unschedule(task_name: str = SCHEDULED_TASK) -> bool:
    """Stop the nightly run. Returns whether there was one to stop."""
    if os.name != "nt":
        raise SystemExit("Scheduling is only set up for Windows.")
    result = _powershell(f"Unregister-ScheduledTask -TaskName {_quoted(task_name)} -Confirm:$false")
    return result.returncode == 0


def scheduled_time(task_name: str = SCHEDULED_TASK) -> str | None:
    """Return when the nightly run is set for, or None if it isn't scheduled."""
    if os.name != "nt":
        return None
    result = _powershell(
        f"(Get-ScheduledTask -TaskName {_quoted(task_name)} -ErrorAction Stop).Triggers[0].StartBoundary"
    )
    found = re.search(r"T(\d\d:\d\d)", result.stdout)
    return found.group(1) if result.returncode == 0 and found else None

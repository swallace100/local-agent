"""The task queue, tested with stand-ins for the runs themselves."""

import json
from dataclasses import replace

import pytest

from agent import jobs
from agent.config import Settings
from agent.jobs import Queue, _only_one_runner, run_queue
from agent.runner import RunResult


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", queue_pause_seconds=0)


@pytest.fixture
def runs(monkeypatch, settings):
    """Replace real runs with ones that succeed, unless the task says to fail."""
    started, resumed = [], []

    async def fake_start(settings, task, project=None, use_web=True, mode="agent", on_start=None, **options):
        run_id = f"run-{len(started) + 1}"
        on_start(run_id)
        started.append({"task": task, "mode": mode, "use_web": use_web, "project": project})
        if "crash" in task:
            raise RuntimeError("the model stopped answering")
        if "ollama" in task:
            raise SystemExit("Ollama isn't running. Start the Ollama app.")
        return RunResult(run_id, "done", settings.runs_dir / run_id)

    async def fake_resume(settings, run_id):
        resumed.append(run_id)
        return RunResult(run_id, "done", settings.runs_dir / run_id)

    monkeypatch.setattr(jobs, "start_run", fake_start)
    monkeypatch.setattr(jobs, "resume_run", fake_resume)
    return started, resumed


def test_the_queue_survives_being_reopened(settings):
    queue = Queue(settings.queue_file)
    first = queue.add("research", "Compare SQLite and DuckDB")
    queue.add("run", "Fix the tests", use_web=False, project="C:/code/app")
    queue.remove(first.id)
    reopened = Queue(settings.queue_file)
    assert [(job.id, job.mode, job.task, job.project) for job in reopened.jobs] == [(2, "run", "Fix the tests", "C:/code/app")]
    assert reopened.add("team", "Another").id == 3
    assert reopened.remove(99) is False


async def test_every_waiting_task_runs_and_one_failure_does_not_stop_the_rest(settings, runs):
    started, _ = runs
    queue = Queue(settings.queue_file)
    queue.add("research", "First question")
    queue.add("research", "This one will crash")
    queue.add("research", "Needs ollama")
    queue.add("run", "Write a poem", use_web=False)

    summary = await run_queue(settings)

    assert [run["task"] for run in started] == ["First question", "This one will crash", "Needs ollama", "Write a poem"]
    assert started[3]["mode"] == "agent", "queue mode 'run' is the general agent"
    after = Queue(settings.queue_file).jobs
    assert [job.status for job in after] == ["done", "failed", "failed", "done"]
    assert "the model stopped answering" in after[1].outcome and "Ollama isn't running" in after[2].outcome
    text = summary.read_text(encoding="utf-8")
    assert "| First question | done |" in text
    assert "failed: the model stopped answering" in text
    assert await run_queue(settings) is None, "nothing is left waiting"


async def test_a_repeating_task_waits_for_the_next_run(settings, runs):
    started, _ = runs
    queue = Queue(settings.queue_file)
    queue.add("moltbook", "agent memory", repeat=True)
    await run_queue(settings)
    await run_queue(settings)
    assert len(started) == 2 and all(run["mode"] == "moltbook" and not run["use_web"] for run in started)
    assert "agent memory" in started[0]["task"] and "digest" in started[0]["task"]
    job = Queue(settings.queue_file).jobs[0]
    assert job.status == "waiting" and job.outcome == "done"


async def test_an_interrupted_task_is_carried_on_first(settings, runs):
    started, resumed = runs
    queue = Queue(settings.queue_file)
    queue.add("research", "Waiting question")
    interrupted = queue.add("research", "Was running when the PC restarted")
    queue.update(interrupted, status="running", run_id="run-old")
    (settings.runs_dir / "run-old").mkdir(parents=True)
    (settings.runs_dir / "run-old" / "run.json").write_text("{}")

    await run_queue(settings)

    assert resumed == ["run-old"]
    assert [run["task"] for run in started] == ["Waiting question"]
    assert all(job.status == "done" for job in Queue(settings.queue_file).jobs)


async def test_an_interrupted_task_whose_run_is_gone_starts_again(settings, runs):
    started, resumed = runs
    queue = Queue(settings.queue_file)
    queue.update(queue.add("research", "Lost its run"), status="running", run_id="run-missing")
    await run_queue(settings)
    assert resumed == [] and [run["task"] for run in started] == ["Lost its run"]


def test_finished_tasks_can_be_cleared(settings):
    queue = Queue(settings.queue_file)
    done, _, failed = queue.add("research", "a"), queue.add("research", "b"), queue.add("research", "c")
    queue.update(done, status="done")
    queue.update(failed, status="failed")
    assert queue.clear_finished() == 2
    assert [job.task for job in Queue(settings.queue_file).jobs] == ["b"]
    assert json.loads(settings.queue_file.read_text(encoding="utf-8"))[0]["status"] == "waiting"


def test_the_queue_can_only_be_worked_through_once_at_a_time(tmp_path):
    with _only_one_runner(tmp_path / "queue.lock"):
        with pytest.raises(SystemExit, match="already being worked through"):
            with _only_one_runner(tmp_path / "queue.lock"):
                pass
    with _only_one_runner(tmp_path / "queue.lock"):
        pass  # Free again once the first has finished.


def test_schedule_times_are_checked(settings, monkeypatch):
    monkeypatch.setattr(jobs.os, "name", "nt")
    for bad in ["2am", "25:00", "12:60", "", "02:00; Remove-Item C:\\"]:
        with pytest.raises(SystemExit, match="24-hour clock"):
            jobs.schedule(settings, bad)

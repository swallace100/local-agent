"""Scores the agent on fixed tasks, and keeps the history so changes can be compared.

Checks run in code and look at what the agent actually produced, not at its own claims.
"""

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from agent.config import ROOT, Settings
from agent.organize import apply_moves
from agent.rag import Knowledge
from agent.runner import start_run

EVALS_DIR = ROOT / "evals"
EVAL_SEARCH_PAUSE_SECONDS = 15.0


def _check(
    check: dict, report: str, run_dir: Path, settings: Settings, folder: Path | None = None
) -> tuple[bool, str]:
    kind = check["type"]
    if kind == "contains":
        return check["text"].lower() in report.lower(), f"report contains {check['text']!r}"
    if kind == "not_contains":
        return check["text"].lower() not in report.lower(), f"report omits {check['text']!r}"
    if kind == "matches":
        found = re.search(check["pattern"], report, re.IGNORECASE)
        return bool(found), f"report matches {check['pattern']!r}"
    if kind == "not_matches":
        found = re.search(check["pattern"], report, re.IGNORECASE)
        return not found, f"report doesn't match {check['pattern']!r}"
    if kind == "min_chars":
        return len(report) >= check["value"], f"report has at least {check['value']} characters"
    if kind == "status":
        status = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["status"]
        return status == check["value"], f"status is {check['value']} (got {status})"
    if kind == "tool_used":
        trace = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
        used = f'"name": "{check["name"]}"' in trace
        return used, f"used tool {check['name']}"
    if kind == "command":
        # Run in the sandbox image so the check can't be affected by this machine.
        workspace = (folder or run_dir / "workspace").resolve()
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none",
             "--env", "PYTHONDONTWRITEBYTECODE=1",
             "--volume", f"{workspace}:/workspace", "--workdir", "/workspace",
             settings.sandbox_image, "sh", "-c", check["command"]],
            capture_output=True,
        )  # fmt: skip
        return result.returncode == 0, f"command passes: {check['command']}"
    raise ValueError(f"Unknown check type: {kind}")


async def run_evals(settings: Settings, only: str | None = None, skip_web: bool = False) -> int:
    tasks = json.loads((EVALS_DIR / "tasks.json").read_text(encoding="utf-8"))
    if skip_web:
        tasks = [t for t in tasks if not t.get("use_web", True)]
    if only:
        tasks = [t for t in tasks if t["id"] == only]
        if not tasks:
            raise SystemExit(f"No eval with ID {only}")

    results = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        # Keep eval knowledge separate from your own.
        knowledge = Knowledge(
            settings, folder=EVALS_DIR / "fixtures" / "knowledge", store_dir=Path(temp) / "chroma"
        )
        knowledge.index()
        # Evals search far more often than normal use, so they wait longer between searches.
        run_settings = replace(
            settings,
            runs_dir=settings.runs_dir / "evals",
            # Evals must not learn from each other, or from your own earlier research.
            remember=False,
            search_pause_seconds=max(settings.search_pause_seconds, EVAL_SEARCH_PAUSE_SECONDS),
        )

        for task in tasks:
            print(f"\n=== {task['id']} ===", flush=True)
            project = EVALS_DIR / "fixtures" / task["project"] if task.get("project") else None
            mode, folder = task.get("mode", "agent"), None
            if mode == "organize":
                # Organize runs plan for a real folder, so give them a copy of the fixture.
                folder, project = Path(temp) / task["id"], None
                shutil.copytree(EVALS_DIR / "fixtures" / task["project"], folder)
            started = time.time()
            try:
                result = await start_run(
                    run_settings,
                    task["task"],
                    project,
                    task.get("use_web", True),
                    knowledge,
                    mode,
                    folder,
                )
                if folder:
                    apply_moves(result.run_dir)
                report = (result.run_dir / "report.md").read_text(encoding="utf-8")
                checks = [_check(c, report, result.run_dir, settings, folder) for c in task["checks"]]
                run_id = result.run_id
            except Exception as error:
                checks = [(False, f"run crashed: {error!r}")]
                run_id = None
            passed = all(ok for ok, _ in checks)
            for ok, description in checks:
                print(f"  {'PASS' if ok else 'FAIL'}  {description}")
            results.append(
                {
                    "id": task["id"],
                    "passed": passed,
                    "seconds": round(time.time() - started),
                    "run_id": run_id,
                    "failed_checks": [d for ok, d in checks if not ok],
                }
            )

    passed_count = sum(r["passed"] for r in results)
    print(f"\nPassed {passed_count} of {len(results)}")
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "commit": _current_commit(),
        "model": settings.model,
        "passed": passed_count,
        "total": len(results),
        "results": results,
    }
    with (EVALS_DIR / "history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(record) + "\n")
    return 0 if passed_count == len(results) else 1


def _current_commit() -> str | None:
    if not shutil.which("git"):
        return None
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() or None

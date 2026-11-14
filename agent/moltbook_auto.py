"""Letting the agent take part on Moltbook without you approving each comment.

This is off until you turn it on with `agent moltbook auto on`, and `auto off` stops it.
A check-in is one pass: answer replies to the agent's own posts, then look for a few posts
on the subjects it cares about and comment where it has something to say.

What stays in force without you there:

- The model still has no tools, notes, files or key. It writes text, and code does the rest.
- Code still refuses any comment with a link, an address, a path, a command or a key.
- The same pace applies: one comment a minute, ten a day.
- Which posts it reads, and how many, is decided by code. Nothing on Moltbook can tell the
  agent what to do: its "what to do next" list and moderators' instructions are never read.
- Three verification puzzles failed in a row turn this off, since ten suspend the account.

Every check-in writes a file in runs/ listing what it read, what it posted and what it
decided against, so you can see how it behaves and tighten it.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import ToolException

from agent.config import ROOT, Settings
from agent.jobs import _powershell, _quoted
from agent.moltbook import POST_ID, POST_URL, _clean, moltbook_tools
from agent.moltbook_account import MIN_SECONDS_BETWEEN_COMMENTS, Account, MoltbookError, Pace
from agent.outbox import draft_comment, solve_challenge

# What the agent looks for. Each check-in searches for a couple of these, in rotation.
# Edit the list to change what it joins in on.
INTERESTS = [
    "how small local models cope with long tasks",
    "checking quotes and citations against sources",
    "agents that verify each other's work",
    "agent memory that can be trusted later",
    "research agents and web search",
]
INTERESTS_PER_CHECKIN = 2
RESULTS_PER_SEARCH = 5
MAX_COMMENTS = 2
# Posts read and considered in one check-in, whether or not the agent comments on them.
MAX_POSTS_CONSIDERED = 6
MAX_REPLIES = 2
MAX_FAILED_PUZZLES = 3
SCHEDULED_TASK = "local-agent moltbook"
LISTED_POST = re.compile(rf"ID: (?P<id>{POST_ID.pattern}) \| m/\S+ \| by (?P<author>.+?) \| ")


@dataclass
class AutoState:
    """Whether the agent may post unprompted, and what it has already answered."""

    path: Path
    on: bool = False
    failed_puzzles: int = 0
    next_interest: int = 0
    commented: list[str] = field(default_factory=list)
    stopped_because: str = ""

    @classmethod
    def load(cls, path: Path) -> "AutoState":
        saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        return cls(path, **saved)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        saved = {name: value for name, value in vars(self).items() if name != "path"}
        self.path.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def auto_state(settings: Settings) -> AutoState:
    return AutoState.load(settings.data_dir / "moltbook" / "auto.json")


@dataclass
class Checkin:
    """A record of one check-in, written up for you afterwards."""

    lines: list[str] = field(default_factory=list)
    posted: int = 0

    def note(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)


async def _send(
    account: Account, model: BaseChatModel, state: AutoState, pace: Pace, post_id: str, text: str, record: Checkin
) -> bool:
    """Post one comment and answer its puzzle. Returns whether it is now public."""
    challenge = account.comment(post_id, text)
    pace.record()
    if challenge:
        # Strict: with nobody to confirm the answer, the numbers must also be found by code.
        solved = await solve_challenge(model, challenge.text, strict=True)
        if not solved:
            state.failed_puzzles += 1
            record.note(
                "  Posted, but the agent couldn't work out Moltbook's puzzle, so the comment stays hidden.\n"
                f"  Puzzle: {challenge.text}"
            )
            return False
        try:
            account.verify(challenge, solved[1])
        except MoltbookError as error:
            state.failed_puzzles += 1
            record.note(
                f"  Posted, but Moltbook rejected the puzzle answer {solved[0]} = {solved[1]:.2f} ({error}).\n"
                f"  Puzzle: {challenge.text}"
            )
            return False
    state.failed_puzzles = 0
    record.posted += 1
    return True


async def _consider(
    account: Account,
    model: BaseChatModel,
    name: str,
    state: AutoState,
    pace: Pace,
    post_id: str,
    record: Checkin,
    dry_run: bool,
    read_post,
) -> None:
    """Read one post, and comment on it if the agent has something to say."""
    try:
        post_text = await read_post(post_id=post_id)
    except ToolException as error:
        record.note(f"- Couldn't read {POST_URL.format(post_id)}: {error}")
        return
    title = next((line[7:] for line in post_text.splitlines() if line.startswith("Title: ")), "")
    record.note(f"- {title}\n  {POST_URL.format(post_id)}")
    text, found = await draft_comment(model, name, post_text, reviewed=False)
    # Remembered either way, so the same post isn't weighed again next time.
    state.commented.append(post_id)
    if text is None:
        record.note("  Nothing to add.")
    elif found:
        record.note(f"  Wrote a comment that was blocked because {', and '.join(found)}:\n  > {text}")
    elif dry_run:
        record.note(f"  Would post:\n  > {text}")
        record.posted += 1
    else:
        if "minute" in (pace.why_not_now() or ""):
            await asyncio.sleep(MIN_SECONDS_BETWEEN_COMMENTS + 5)
        if pace.why_not_now():
            record.note(f"  Not posted: {pace.why_not_now()}")
            return
        if await _send(account, model, state, pace, post_id, text, record):
            record.note(f"  Posted:\n  > {text}")


async def check_in(
    settings: Settings,
    model: BaseChatModel,
    account: Account,
    name: str,
    dry_run: bool = False,
    max_comments: int = MAX_COMMENTS,
    reader_tools=None,
) -> Path:
    """Run one check-in and return the path of its write-up."""
    state = auto_state(settings)
    if not state.on and not dry_run:
        raise SystemExit("Posting without approval is off. Turn it on with: agent moltbook auto on")
    pace = Pace(settings.data_dir / "moltbook" / "pace.json")
    tools = {tool.name: tool.coroutine for tool in (reader_tools or moltbook_tools())}
    record = Checkin()
    started = datetime.now()
    record.note(f"# Moltbook check-in, {started:%d %B %Y %H:%M}" + (" (dry run, nothing posted)" if dry_run else ""))

    try:
        # Replies to the agent's own posts come first. Only the list of posts is taken from
        # Moltbook's dashboard. Its suggestions and instructions are never read.
        activity = (account.home().get("activity_on_your_posts") or [])[:MAX_REPLIES]
        if activity:
            record.note("\n## Replies to the agent's posts\n")
        for entry in activity:
            post_id = _clean(entry.get("post_id"), 40)
            if not POST_ID.fullmatch(post_id) or record.posted >= max_comments:
                continue
            state.commented = [seen for seen in state.commented if seen != post_id]
            await _consider(account, model, name, state, pace, post_id, record, dry_run, tools["moltbook_read_post"])
            if not dry_run:
                account.mark_read(post_id)

        failures_before = state.failed_puzzles
        record.note("\n## Posts on the agent's subjects\n")
        candidates: list[str] = []
        for _ in range(INTERESTS_PER_CHECKIN):
            interest = INTERESTS[state.next_interest % len(INTERESTS)]
            state.next_interest += 1
            try:
                listing = await tools["moltbook_search"](query=interest, limit=RESULTS_PER_SEARCH)
            except ToolException as error:
                record.note(f"Couldn't search for \"{interest}\": {error}")
                continue
            for found in LISTED_POST.finditer(listing):
                post_id, author = found["id"], found["author"]
                if post_id not in state.commented and post_id not in candidates and author.strip() != name:
                    candidates.append(post_id)
        for post_id in candidates[:MAX_POSTS_CONSIDERED]:
            # One failed puzzle ends the check-in. Each failure counts towards a suspension,
            # so the agent doesn't try its luck on another post.
            if record.posted >= max_comments or state.failed_puzzles > failures_before:
                break
            await _consider(account, model, name, state, pace, post_id, record, dry_run, tools["moltbook_read_post"])
        if not candidates:
            record.note("Found nothing new to read.")
    except MoltbookError as error:
        record.note(f"\nStopped early: {error}")

    if state.failed_puzzles >= MAX_FAILED_PUZZLES:
        state.on = False
        state.stopped_because = f"{MAX_FAILED_PUZZLES} verification puzzles failed in a row."
        record.note(f"\nPosting without approval has been turned off: {state.stopped_because}")
    state.commented = state.commented[-500:]
    if not dry_run:
        state.save()
    done = "would have posted" if dry_run else "posted"
    record.note(f"\nThe agent {done} {record.posted} comments.")
    path = settings.runs_dir / f"moltbook-checkin-{started:%Y%m%d-%H%M%S}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(record.lines) + "\n", encoding="utf-8")
    return path


def schedule_checkins(settings: Settings, hours: int, task_name: str = SCHEDULED_TASK) -> None:
    """Have Windows run a check-in every so many hours."""
    if os.name != "nt":
        raise SystemExit("Scheduling is only set up for Windows. Elsewhere, add `agent moltbook checkin` to cron.")
    if not 1 <= hours <= 168:
        raise SystemExit("Give the gap between check-ins in hours, from 1 to 168.")
    program = Path(sys.executable).with_name("agent.exe")
    log = settings.data_dir / "moltbook" / "checkin.log"
    command = f'moltbook checkin --log "{log}"'
    script = (
        f"$action = New-ScheduledTaskAction -Execute {_quoted(program)} "
        f"-Argument {_quoted(command)} -WorkingDirectory {_quoted(ROOT)}; "
        "$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(10) "
        f"-RepetitionInterval (New-TimeSpan -Hours {hours}); "
        "$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1); "
        f"Register-ScheduledTask -TaskName {_quoted(task_name)} -Action $action -Trigger $trigger "
        "-Settings $settings -Force | Out-Null"
    )
    result = _powershell(script)
    if result.returncode != 0:
        raise SystemExit(f"Windows couldn't schedule the check-ins: {result.stderr.strip()}")


def unschedule_checkins(task_name: str = SCHEDULED_TASK) -> bool:
    if os.name != "nt":
        return False
    return _powershell(f"Unregister-ScheduledTask -TaskName {_quoted(task_name)} -Confirm:$false").returncode == 0


def checkins_scheduled(task_name: str = SCHEDULED_TASK) -> bool:
    if os.name != "nt":
        return False
    return _powershell(f"Get-ScheduledTask -TaskName {_quoted(task_name)} -ErrorAction Stop").returncode == 0

"""The `agent moltbook ...` commands for the agent's own account: join, draft, approve.

Each one is started by you at the command line. Nothing here runs on a schedule, and no
model output can start any of it.
"""

import json
from collections.abc import Callable
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.config import Settings
from agent.moltbook import POST_URL, _clean, moltbook_tools
from agent.moltbook_account import (
    CREDENTIALS_FILE,
    Account,
    Challenge,
    MoltbookError,
    Pace,
    claim_details,
    load_key,
)
from agent.outbox import (
    MAX_COMMENT_CHARS,
    MAX_POST_CHARS,
    Outbox,
    draft_comment,
    draft_post,
    post_id_from,
    problems,
    reading_checks_out,
    solve_challenge,
    tidy_challenge,
    title_problems,
)

PREVIEW_CHARS = 200


def _outbox(settings: Settings) -> Outbox:
    return Outbox(settings.data_dir / "moltbook" / "outbox.json")


def _account(credentials: Path) -> tuple[Account, str]:
    saved = load_key(credentials)
    if not saved:
        raise SystemExit('The agent hasn\'t joined Moltbook yet. Run: agent moltbook join "Name" "Description"')
    return Account(saved[0]), saved[1]


def join(name: str, description: str, credentials: Path = CREDENTIALS_FILE, account: Account | None = None) -> None:
    """Register the agent and tell you how to claim it. The key is saved, never shown."""
    try:
        registered = (account or Account()).register(name, description, credentials)
    except MoltbookError as error:
        raise SystemExit(f"Couldn't join Moltbook: {error}") from error
    print(f"{registered.name} is registered. Its key is saved in {credentials}")
    print("\nTo finish, claim the agent yourself. Moltbook will ask you to verify an email")
    print("address and post a tweet containing the code below.")
    print(f"\n  Claim link:        {registered.claim_url}")
    print(f"  Verification code: {registered.verification_code}")
    print("\nThen check with: agent moltbook status")


def status(credentials: Path = CREDENTIALS_FILE, account: Account | None = None) -> None:
    saved = load_key(credentials)
    if not saved:
        print("The agent hasn't joined Moltbook.")
        return
    try:
        state = (account or Account(saved[0])).status()
    except MoltbookError as error:
        raise SystemExit(f"Couldn't check: {error}") from error
    if state == "claimed":
        print(f"{saved[1]} is claimed and can post what you approve. Profile: https://www.moltbook.com/u/{saved[1]}")
    else:
        claim_url, code = claim_details(credentials)
        print(f"{saved[1]} is registered but not claimed yet ({state}).")
        print(f"  Claim link:        {claim_url}")
        print(f"  Verification code: {code}")


async def draft(
    settings: Settings,
    model: BaseChatModel,
    name: str,
    reference: str | None,
    new_post: str | None,
    submolt: str,
    reply_to: str | None,
) -> None:
    """Have the model write a draft, check it, and put it in the outbox."""
    outbox = _outbox(settings)
    if new_post:
        title, body, found = await draft_post(model, name, new_post)
        if found:
            print(f"The draft was not saved, because {', and '.join(found)}:\n\n{title}\n\n{body}")
            return
        saved = outbox.add("post", body, submolt=submolt, title=title)
    else:
        post_id = post_id_from(reference or "")
        if not post_id:
            raise SystemExit("Give the post's ID or its moltbook.com address, or use --new to draft a post.")
        # Read without the account, through the same cleaned-up reader the digest uses.
        read_post = next(tool for tool in moltbook_tools() if tool.name == "moltbook_read_post")
        post_text = await read_post.coroutine(post_id=post_id)
        text, found = await draft_comment(model, name, post_text)
        if text is None:
            print("The agent had nothing specific to add to that post, so there is no draft.")
            return
        if found:
            print(f"The draft was not saved, because {', and '.join(found)}:\n\n{text}")
            return
        title = next((line[7:] for line in post_text.splitlines() if line.startswith("Title: ")), "")
        saved = outbox.add("comment", text, post_id=post_id, post_title=title, parent_id=reply_to or "")
    print(f"Draft {saved.id} is in the outbox. Nothing has been posted.\n")
    show(settings, only=saved.id)


def show(settings: Settings, only: int | None = None) -> None:
    drafts = [d for d in _outbox(settings).waiting() if only is None or d.id == only]
    if not drafts:
        print("The outbox is empty.")
        return
    for item in drafts:
        if item.kind == "post":
            print(f"[{item.id}] New post in m/{item.submolt}: {item.title}")
        else:
            reply = " (a reply to another comment)" if item.parent_id else ""
            print(f"[{item.id}] Comment{reply} on: {item.post_title}\n    {POST_URL.format(item.post_id)}")
        print(f"\n{item.text}\n")
        print(f"    Post it:  agent moltbook approve {item.id}      Drop it:  agent moltbook reject {item.id}\n")


def reject(settings: Settings, draft_id: int) -> None:
    outbox = _outbox(settings)
    item = outbox.get(draft_id)
    if not item or item.status != "draft":
        raise SystemExit(f"There is no draft {draft_id} waiting.")
    outbox.update(item, status="rejected")
    print(f"Draft {draft_id} dropped.")


def _confirm_answer(question: str) -> str:
    return input(question)


async def approve(
    settings: Settings,
    model: BaseChatModel,
    draft_id: int,
    edited: str | None = None,
    account: Account | None = None,
    confirm: Callable[[str], str] = _confirm_answer,
    credentials: Path = CREDENTIALS_FILE,
    answer_later: bool = False,
) -> None:
    """Send one draft. With answer_later, the puzzle is saved for `agent moltbook answer`."""
    outbox = _outbox(settings)
    item = outbox.get(draft_id)
    if not item or item.status != "draft":
        raise SystemExit(f"There is no draft {draft_id} waiting.")
    text = edited if edited is not None else item.text
    # Checked again here, because an edit or a hand-changed outbox file skips the first check.
    found = problems(text, MAX_POST_CHARS if item.kind == "post" else MAX_COMMENT_CHARS)
    found += title_problems(item.title) if item.kind == "post" else []
    if found:
        raise SystemExit(f"Not sent, because {', and '.join(found)}.")
    pace = Pace(settings.data_dir / "moltbook" / "pace.json")
    if pace.why_not_now():
        raise SystemExit(pace.why_not_now())
    account = account or _account(credentials)[0]

    try:
        if account.status() != "claimed":
            raise SystemExit("The agent's account hasn't been claimed yet, so it can't post. See: agent moltbook status")
        if item.kind == "post":
            challenge = account.post(item.submolt, item.title, text)
        else:
            challenge = account.comment(item.post_id, text, item.parent_id or None)
        pace.record()
        if challenge:
            # Moltbook hides the text until a puzzle is solved. Ten wrong answers in a row
            # suspend the account, so you see the answer before it is sent.
            solved = await solve_challenge(model, challenge.text)
            print(f"\nMoltbook set a puzzle before it will show this:\n  {challenge.text}")
            print(f"With the symbols taken out:\n  {tidy_challenge(challenge.text)}")
            if answer_later:
                # For when nobody is at the keyboard to answer straight away.
                if solved:
                    print(f"The agent reads that as: {solved[0]} = {solved[1]:.2f}")
                pending = {"draft": item.id, "code": challenge.code, "text": challenge.text}
                _pending_file(settings).write_text(json.dumps(pending), encoding="utf-8")
                outbox.update(item, text=text, status="pending", note="Waiting for the puzzle's answer.")
                print("It stays hidden until answered, within 5 minutes: agent moltbook answer <number>")
                return
            if solved:
                print(f"The agent reads that as: {solved[0]} = {solved[1]:.2f}")
                if not reading_checks_out(challenge.text, solved[0]):
                    print("Check it carefully: those aren't the numbers code found in the puzzle.")
                reply = confirm("Press Enter to send that answer, type a different number, or type n to give up: ")
            else:
                print("The agent couldn't work it out.")
                reply = confirm("Type the answer as a number, or n to give up: ")
            reply = reply.strip()
            if reply.lower() == "n" or (not reply and not solved):
                outbox.update(item, text=text, status="failed", note="The puzzle wasn't answered, so Moltbook is hiding it.")
                print("Left unanswered. Moltbook will keep the text hidden.")
                return
            account.verify(challenge, float(reply) if reply else solved[1])
    except MoltbookError as error:
        outbox.update(item, text=text, status="failed", note=str(error))
        raise SystemExit(f"Moltbook didn't accept it: {error}") from error
    except ValueError:
        outbox.update(item, text=text, status="failed", note="The answer given wasn't a number.")
        raise SystemExit("That wasn't a number, so no answer was sent. Moltbook will keep the text hidden.") from None
    outbox.update(item, text=text, status="sent")
    print(f"Draft {draft_id} is posted.")


def _pending_file(settings: Settings) -> Path:
    return settings.data_dir / "moltbook" / "pending-puzzle.json"


def answer(
    settings: Settings, number: float, account: Account | None = None, credentials: Path = CREDENTIALS_FILE
) -> None:
    """Send your answer to the puzzle that `approve --answer-later` left waiting."""
    pending_file = _pending_file(settings)
    if not pending_file.is_file():
        raise SystemExit("There is no puzzle waiting for an answer.")
    pending = json.loads(pending_file.read_text(encoding="utf-8"))
    outbox = _outbox(settings)
    item = outbox.get(pending["draft"])
    pending_file.unlink()
    try:
        (account or _account(credentials)[0]).verify(Challenge(pending["code"], pending["text"]), number)
    except MoltbookError as error:
        if item:
            outbox.update(item, status="failed", note=str(error))
        raise SystemExit(f"Moltbook didn't accept the answer: {error}") from error
    if item:
        outbox.update(item, status="sent", note="")
    print(f"Accepted. Draft {pending['draft']} is posted.")


def inbox(account: Account | None = None, credentials: Path = CREDENTIALS_FILE) -> None:
    """Show what has happened on the agent's posts.

    Moltbook's reply also says what the agent should do next and can carry instructions
    from community moderators. Those parts are left out: they are not yours.
    """
    account = account or _account(credentials)[0]
    try:
        home = account.home()
    except MoltbookError as error:
        raise SystemExit(f"Couldn't check: {error}") from error
    you = home.get("your_account") or {}
    print(f"{_clean(you.get('name'), 60)}: karma {you.get('karma', 0)}, {you.get('unread_notification_count', 0)} unread")
    activity = home.get("activity_on_your_posts") or []
    if not activity:
        print("Nothing new on the agent's posts.")
    for entry in activity:
        who = ", ".join(_clean(name, 60) for name in (entry.get("latest_commenters") or [])[:5])
        print(f"\n{entry.get('new_notification_count', 0)} new on: {_clean(entry.get('post_title'), PREVIEW_CHARS)}")
        print(f"  from: {who or 'unknown'}")
        print(f"  post: {_clean(entry.get('post_id'), 40)}")
        print(f"  To draft a reply: agent moltbook draft {_clean(entry.get('post_id'), 40)}")

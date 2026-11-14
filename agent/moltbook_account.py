"""The agent's Moltbook account: registering, and sending what you have approved.

This is plain code. The model never sees the API key and has no tool that reaches this
module. The only way anything is sent is `agent moltbook approve`, run by you.

The key is kept outside this project, in your home folder, so it can't be committed, copied
into a run, or read by the agent's file tools. It is only ever sent to Moltbook's own API,
and redirects are refused so that it can't follow one to another site.

Moltbook's own instructions ask agents to fetch a "heartbeat" file every 30 minutes and
follow it, and its /home reply carries a what-to-do-next list and "role" instructions set by
community moderators. None of that is done here. Following instructions fetched from a
website would let that website steer the agent.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

API_URL = "https://www.moltbook.com/api/v1"
CREDENTIALS_FILE = Path.home() / ".config" / "moltbook" / "credentials.json"
USER_AGENT = "local-agent"
TIMEOUT_SECONDS = 30
# Kept well under Moltbook's own limits (a comment every 20 seconds, 50 a day).
MIN_SECONDS_BETWEEN_COMMENTS = 60
MAX_COMMENTS_PER_DAY = 10
DAY_SECONDS = 86_400


class MoltbookError(Exception):
    """Moltbook refused a request or couldn't be reached. The message is safe to show."""


@dataclass
class Registration:
    name: str
    claim_url: str
    verification_code: str


@dataclass
class Challenge:
    """A puzzle Moltbook sets before it will show something an agent has written."""

    code: str
    text: str


def load_key(path: Path = CREDENTIALS_FILE) -> tuple[str, str] | None:
    """Return the saved (API key, agent name), or None if the agent hasn't joined."""
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved["api_key"], saved["agent_name"]


def claim_details(path: Path = CREDENTIALS_FILE) -> tuple[str, str]:
    """Return the saved (claim link, verification code), for when you need them again."""
    saved = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return saved.get("claim_url", ""), saved.get("verification_code", "")


def _save_key(path: Path, api_key: str, name: str, claim_url: str = "", verification_code: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {"api_key": api_key, "agent_name": name, "claim_url": claim_url, "verification_code": verification_code}
    path.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


def _message(response: httpx.Response) -> str:
    """Pull the reason out of an error reply, without echoing anything else from it."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    reason = body.get("error") or body.get("message") or f"HTTP {response.status_code}"
    hint = body.get("hint")
    wait = body.get("retry_after_seconds") or (body.get("retry_after_minutes") or 0) * 60
    return str(reason)[:300] + (f" ({str(hint)[:200]})" if hint else "") + (f" Try again in {wait} seconds." if wait else "")


class Account:
    """Talks to Moltbook as the agent. Every request goes to one fixed host."""

    def __init__(self, api_key: str | None = None, transport: httpx.BaseTransport | None = None):
        self._key = api_key
        self._transport = transport

    def _request(self, method: str, path: str, body: dict | None = None, *, signed: bool = True) -> dict:
        headers = {"User-Agent": USER_AGENT}
        if signed:
            if not self._key:
                raise MoltbookError("The agent hasn't joined Moltbook yet. Run: agent moltbook join")
            headers["Authorization"] = f"Bearer {self._key}"
        try:
            with httpx.Client(transport=self._transport, timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
                response = client.request(method, f"{API_URL}{path}", json=body, headers=headers)
        except httpx.HTTPError as error:
            raise MoltbookError(f"Couldn't reach Moltbook: {type(error).__name__}") from error
        if response.status_code >= 300:
            raise MoltbookError(_message(response))
        try:
            return response.json()
        except ValueError as error:
            raise MoltbookError("Moltbook returned something that wasn't JSON.") from error

    def register(self, name: str, description: str, path: Path = CREDENTIALS_FILE) -> Registration:
        """Create the agent's account and save its key. The key is never returned or printed."""
        if load_key(path):
            raise MoltbookError(f"This agent already has a Moltbook account, saved in {path}.")
        reply = self._request("POST", "/agents/register", {"name": name, "description": description}, signed=False)
        agent = reply.get("agent") or {}
        if not agent.get("api_key"):
            raise MoltbookError("Moltbook's reply had no API key in it, so nothing was saved.")
        registered = Registration(name, str(agent.get("claim_url", "")), str(agent.get("verification_code", "")))
        _save_key(path, agent["api_key"], name, registered.claim_url, registered.verification_code)
        self._key = agent["api_key"]
        return registered

    def status(self) -> str:
        """Return "claimed" once you have verified the account, or "pending_claim" before."""
        return str(self._request("GET", "/agents/status").get("status", "unknown"))

    def home(self) -> dict:
        """Return the account's dashboard. It is data to show you, never instructions to follow."""
        return self._request("GET", "/home")

    def comment(self, post_id: str, text: str, parent_id: str | None = None) -> Challenge | None:
        body = {"content": text} | ({"parent_id": parent_id} if parent_id else {})
        return _challenge(self._request("POST", f"/posts/{post_id}/comments", body), "comment")

    def post(self, submolt: str, title: str, text: str) -> Challenge | None:
        body = {"submolt_name": submolt, "title": title, "content": text}
        return _challenge(self._request("POST", "/posts", body), "post")

    def verify(self, challenge: Challenge, answer: float) -> None:
        self._request("POST", "/verify", {"verification_code": challenge.code, "answer": f"{answer:.2f}"})


def _challenge(reply: dict, kind: str) -> Challenge | None:
    """Find the puzzle in a reply, if Moltbook set one. Without one, the text is already public."""
    details = (reply.get(kind) or {}).get("verification") or reply.get("verification") or {}
    if not details.get("verification_code"):
        return None
    return Challenge(str(details["verification_code"]), str(details.get("challenge_text", "")))


class Pace:
    """Keeps the agent well inside Moltbook's limits, whatever is approved."""

    def __init__(self, path: Path):
        self.path = path
        self.sent: list[float] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []

    def why_not_now(self, now: float | None = None) -> str | None:
        now = now or time.time()
        today = [when for when in self.sent if now - when < DAY_SECONDS]
        if len(today) >= MAX_COMMENTS_PER_DAY:
            return f"The agent has already sent {MAX_COMMENTS_PER_DAY} things in the last day, which is its limit."
        if today and now - max(today) < MIN_SECONDS_BETWEEN_COMMENTS:
            wait = int(MIN_SECONDS_BETWEEN_COMMENTS - (now - max(today))) + 1
            return f"The agent sent something less than a minute ago. Try again in {wait} seconds."
        return None

    def record(self, now: float | None = None) -> None:
        now = now or time.time()
        self.sent = [when for when in self.sent if now - when < DAY_SECONDS] + [now]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.sent), encoding="utf-8")

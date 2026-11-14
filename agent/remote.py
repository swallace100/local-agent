"""Letting agents on other machines join a team run.

    Host:    agent team "question" --host
    Others:  agent team join http://<host>:8765 --key <key>

The host serves its board over the network. A joining agent uses its own model, GPU and
web access to do tasks from that board, so the team really does work in parallel.

An agent on another machine is trusted less than one on this machine, and the host
enforces that in code:

- Every request needs the key the host printed.
- A joiner's notes are kept only if the host can read the page itself and find the quote
  there. A joiner can't put words on the board that no page says.
- Your private notes in knowledge/ never leave the host. Joiners see the sub-questions and
  the notes that came from web pages, and nothing else.
- A joiner can only finish or give up a task it holds, and can't act under another's name.
- Free text from a joiner (a reworded question, a verdict's reason) is cut short and cleaned.

The connection is plain HTTP, so anyone on the network path could read it. Use it on a
network you trust, such as your home network or a private VPN, never across the open internet.
"""

import asyncio
import hmac
import json
import re
import secrets
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from agent.moltbook import HIDDEN_CHARS
from agent.research import Findings, parse_page, quote_in_source
from agent.team import VERDICTS, Board, Task

DEFAULT_PORT = 8765
MAX_BODY_BYTES = 256_000
MAX_NOTES_PER_TASK = 12
MAX_URLS_PER_TASK = 20
MAX_TEXT_CHARS = 200
SECTIONS_TO_CHECK = 2
VERIFY_TIMEOUT_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 120
NAME = re.compile(r"[^A-Za-z0-9_-]")
WEB_SOURCE = ("http://", "https://")
ALLOWED_VERDICTS = (*VERDICTS, "UNCONFIRMED")


class BoardGone(Exception):
    """The host has stopped serving its board, usually because the run finished."""


def _short(text: object) -> str:
    return " ".join(HIDDEN_CHARS.sub("", str(text or "")).split())[:MAX_TEXT_CHARS]


def remote_name(name: object) -> str:
    """Joiners are always labelled as such, so they can't pass as an agent on the host."""
    cleaned = NAME.sub("", str(name or "")).removeprefix("remote-")[:24]
    return "remote-" + (cleaned or "agent")


class BoardServer:
    """Serves one board to agents on other machines, checking everything they send."""

    def __init__(
        self,
        board_path: Path,
        questions: list[str],
        read_page: Callable[[str, int], Awaitable[str]],
        loop: asyncio.AbstractEventLoop,
        lease_seconds: float,
        max_attempts: int,
        listen: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
    ):
        self.key = secrets.token_urlsafe(18)
        self.questions = questions
        self.joined: set[str] = set()
        self._board = Board(board_path, lease_seconds=lease_seconds, any_thread=True)
        self._lock = threading.Lock()
        self._read_page, self._loop, self._max_attempts = read_page, loop, max_attempts
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # Keep the run's own output readable.
                pass

            def do_POST(self) -> None:
                status, reply = server._handle(self)
                body = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._http = ThreadingHTTPServer((listen, port), Handler)
        self.port = self._http.server_address[1]
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()
        self._board.close()

    def _handle(self, request: BaseHTTPRequestHandler) -> tuple[int, dict]:
        given = request.headers.get("Authorization", "")
        if not hmac.compare_digest(given.encode(), f"Bearer {self.key}".encode()):
            return 401, {"error": "The key is wrong or missing."}
        try:
            length = int(request.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY_BYTES:
                return 413, {"error": "The request is empty or too large."}
            call = json.loads(request.rfile.read(length))
            method = getattr(self, f"_do_{call['method']}", None)
            if method is None:
                return 404, {"error": "There is no such request."}
            agent = remote_name(call.get("agent"))
            self.joined.add(agent)
            return 200, {"result": method(agent, **(call.get("args") or {}))}
        except (ValueError, TypeError, KeyError, LookupError) as error:
            return 400, {"error": f"The request wasn't understood: {type(error).__name__}"}

    def _question(self, index: object) -> int:
        if not isinstance(index, int) or not 0 <= index < len(self.questions):
            raise ValueError("no such question")
        return index

    def _own_task(self, agent: str, task_id: object) -> Task:
        """Return a task this agent holds. Anything else is refused."""
        with self._lock:
            if not isinstance(task_id, int) or self._board.held_by(task_id) != agent:
                raise LookupError("not this agent's task")
        return Task(task_id, "", 0, None, 1)

    # Each _do_ method is one request a joiner may make. Nothing else is reachable.

    def _do_questions(self, agent: str) -> list[str]:
        return self.questions

    def _do_claim(self, agent: str) -> dict | None:
        with self._lock:
            task = self._board.claim(agent)
        return asdict(task) if task else None

    def _do_finish(self, agent: str, task: int) -> None:
        owned = self._own_task(agent, task)
        with self._lock:
            self._board.finish(owned)

    def _do_release(self, agent: str, task: int) -> None:
        owned = self._own_task(agent, task)
        with self._lock:
            self._board.release(owned)

    def _do_in_progress(self, agent: str) -> int:
        with self._lock:
            return self._board.in_progress()

    def _do_unfinished(self, agent: str) -> int:
        with self._lock:
            return self._board.unfinished()

    def _do_opened(self, agent: str, question: int) -> list[str]:
        with self._lock:
            return sorted(self._board.opened(self._question(question)))

    def _do_notes(self, agent: str, question: int) -> list[dict]:
        """Notes from web pages only. Notes from the host's private files are never sent."""
        with self._lock:
            notes = self._board.notes(self._question(question))
        return [note for note in notes if note["source"].startswith(WEB_SOURCE)]

    def _do_record(self, agent: str, question: int, notes: list, opened: list, failed_searches: int = 0) -> dict:
        """Take a joiner's findings, keeping only the notes the host can confirm for itself."""
        index = self._question(question)
        candidates = [
            {"question": index, "source": str(n["source"]), "quote": str(n["quote"])[:400], "fact": _short(n["fact"])}
            for n in list(notes)[:MAX_NOTES_PER_TASK]
            if str(n.get("source", "")).startswith(WEB_SOURCE)
        ]
        future = asyncio.run_coroutine_threadsafe(self._confirm(candidates), self._loop)
        kept = future.result(timeout=VERIFY_TIMEOUT_SECONDS)
        found = Findings(
            notes=kept,
            dropped=len(list(notes)) - len(kept),
            failed_searches=max(0, min(int(failed_searches), 10)),
            opened=[str(url) for url in list(opened)[:MAX_URLS_PER_TASK] if str(url).startswith(WEB_SOURCE)],
        )
        with self._lock:
            self._board.record(agent, index, found)
        return {"kept": len(kept), "dropped": found.dropped}

    async def _confirm(self, notes: list[dict]) -> list[dict]:
        """Read each page here on the host, and keep the notes whose quotes are really in it."""
        pages: dict[str, str] = {}
        kept = []
        for note in notes:
            url = note["source"]
            if url not in pages:
                pages[url] = await self._page_text(url)
            if quote_in_source(note["quote"], pages[url]):
                kept.append(note)
        return kept

    async def _page_text(self, url: str) -> str:
        text, offset = "", 0
        for _ in range(SECTIONS_TO_CHECK):
            try:
                _, section, next_offset = parse_page(await self._read_page(url, offset), url)
            except Exception:  # A page the host can't read can't confirm anything.
                break
            text += "\n" + section
            if next_offset is None:
                break
            offset = next_offset
        return text

    def _do_post(self, agent: str, kind: str, question: int, query: str | None = None, attempt: int = 1) -> None:
        if kind not in ("research", "check"):
            raise ValueError("no such kind of task")
        attempt = int(attempt)
        if not 1 <= attempt <= self._max_attempts:
            raise ValueError("too many attempts")
        with self._lock:
            # Always marked as not for the sender, so a joiner can't hand itself a check.
            self._board.post(kind, self._question(question), not_for=agent, query=_short(query) or None, attempt=attempt)

    def _do_set_check(self, agent: str, question: int, verdict: str, reason: str = "") -> None:
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError("no such verdict")
        with self._lock:
            self._board.set_check(self._question(question), agent, verdict, _short(reason))

    def _do_say(self, agent: str, text: str) -> None:
        with self._lock:
            self._board.say(agent, _short(text))


class RemoteBoard:
    """A joiner's view of the host's board. It has the same methods as Board, so the same
    agent code runs whether the board is a local file or on another machine."""

    def __init__(self, url: str, key: str, name: str, transport: httpx.BaseTransport | None = None):
        self.name = remote_name(name)
        self._url = url.rstrip("/") + "/rpc"
        self._client = httpx.Client(
            transport=transport, timeout=REQUEST_TIMEOUT_SECONDS, headers={"Authorization": f"Bearer {key}"}
        )

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, **args):
        try:
            response = self._client.post(self._url, json={"method": method, "agent": self.name, "args": args})
        except httpx.HTTPError as error:
            raise BoardGone(f"The host can't be reached: {type(error).__name__}") from error
        if response.status_code == 401:
            raise SystemExit("The host refused the key. Copy it again from the host's screen.")
        if response.status_code != 200:
            raise BoardGone(response.json().get("error", f"HTTP {response.status_code}"))
        return response.json()["result"]

    def questions(self) -> list[str]:
        return self._call("questions")

    def claim(self, agent: str) -> Task | None:
        task = self._call("claim")
        return Task(**task) if task else None

    def finish(self, task: Task) -> None:
        self._call("finish", task=task.id)

    def release(self, task: Task) -> None:
        self._call("release", task=task.id)

    def in_progress(self) -> int:
        return self._call("in_progress")

    def unfinished(self) -> int:
        return self._call("unfinished")

    def opened(self, question: int) -> set[str]:
        return set(self._call("opened", question=question))

    def notes(self, question: int | None = None) -> list[dict]:
        return self._call("notes", question=question)

    def record(self, agent: str, question: int, found: Findings) -> None:
        result = self._call(
            "record", question=question, notes=found.notes, opened=found.opened, failed_searches=found.failed_searches
        )
        if result["dropped"]:
            print(f"  The host couldn't confirm {result['dropped']} of those notes, and left them out.", flush=True)

    def post(self, kind: str, question: int, *, not_for: str, query: str | None = None, attempt: int = 1) -> None:
        self._call("post", kind=kind, question=question, query=query, attempt=attempt)

    def set_check(self, question: int, agent: str, verdict: str, reason: str) -> None:
        self._call("set_check", question=question, verdict=verdict, reason=reason)

    def say(self, agent: str, text: str) -> None:
        print(f"{self.name}: {text}", flush=True)
        self._call("say", text=text)

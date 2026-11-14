"""Agents on other machines joining a board, tested with a real server on this machine."""

import asyncio
import threading

import httpx
import pytest
from test_team import PAGES, RESULTS, RuleModel, web

from agent.config import Settings
from agent.remote import BoardGone, BoardServer, RemoteBoard, remote_name
from agent.research import Findings, Researcher, build_research_graph, initial_research_state
from agent.team import Board, _work, team_stage

SETTINGS = Settings()
QUESTIONS = ["When was SQLite released?", "When was DuckDB released?"]
REAL_NOTE = {"question": 0, "source": "https://a.org/sqlite", "quote": "SQLite was first released in August 2000.", "fact": "2000"}


async def host_reads(url: str, offset: int = 0) -> str:
    """What the host sees when it opens a page to check a joiner's note."""
    if url not in PAGES:
        raise RuntimeError("no such page")
    return f"Title: T\nURL: {url}\n\n{PAGES[url]}"


@pytest.fixture
def loop():
    """An event loop on its own thread, standing in for the host's running run."""
    running = asyncio.new_event_loop()
    thread = threading.Thread(target=running.run_forever, daemon=True)
    thread.start()
    yield running
    running.call_soon_threadsafe(running.stop)
    thread.join(timeout=5)


@pytest.fixture
def served(tmp_path, loop):
    path = tmp_path / "board.sqlite"
    local = Board(path)
    local.start(len(QUESTIONS))
    server = BoardServer(path, QUESTIONS, host_reads, loop, lease_seconds=900, max_attempts=2, listen="127.0.0.1", port=0)
    yield server, local, f"http://127.0.0.1:{server.port}"
    server.close()
    local.close()


def test_nothing_is_served_without_the_key(served):
    server, _, url = served
    assert httpx.post(f"{url}/rpc", json={"method": "questions"}).status_code == 401
    with pytest.raises(SystemExit, match="refused the key"):
        RemoteBoard(url, "wrong-key", "laptop").questions()
    assert RemoteBoard(url, server.key, "laptop").questions() == QUESTIONS
    assert httpx.post(f"{url}/rpc", json={"method": "_handle"}, headers={"Authorization": f"Bearer {server.key}"}).status_code == 404


def test_a_joiner_is_always_labelled_and_can_only_finish_its_own_tasks(served):
    server, local, url = served
    joiner = RemoteBoard(url, server.key, "agent-1")
    assert joiner.name == "remote-agent-1" and remote_name("../x y") == "remote-xy"
    mine = joiner.claim(joiner.name)
    theirs = local.claim("agent-1")
    assert local.held_by(mine.id) == "remote-agent-1"
    with pytest.raises(BoardGone):
        joiner.finish(theirs)
    assert local.held_by(theirs.id) == "agent-1", "the host's own task is untouched"
    joiner.finish(mine)
    assert local.unfinished() == 1


def test_only_notes_the_host_can_confirm_are_kept(served):
    server, local, url = served
    joiner = RemoteBoard(url, server.key, "laptop")
    found = Findings(
        notes=[
            REAL_NOTE,
            {"question": 0, "source": "https://a.org/sqlite", "quote": "SQLite was written by aliens in 1987.", "fact": "made up"},
            {"question": 0, "source": "https://nowhere.example/page", "quote": "A page the host cannot open at all.", "fact": "unreachable"},
            {"question": 0, "source": "knowledge/secrets.md", "quote": "SQLite was first released in August 2000.", "fact": "not a web page"},
        ],
        opened=["https://a.org/sqlite", "file:///C:/Users/me/secret.txt"],
    )
    joiner.record(joiner.name, 0, found)
    assert [(note["quote"], note["agent"]) for note in local.notes(0)] == [(REAL_NOTE["quote"], "remote-laptop")]
    assert local.count("dropped") == 3
    assert local.opened(0) == {"https://a.org/sqlite"}


def test_private_notes_are_never_sent_to_a_joiner(served):
    server, local, url = served
    private = {"question": 0, "source": "knowledge/salaries.md", "quote": "Marisol earns a great deal.", "fact": "private"}
    local.record("agent-1", 0, Findings(notes=[private, REAL_NOTE]))
    seen = RemoteBoard(url, server.key, "laptop").notes(0)
    assert [note["source"] for note in seen] == ["https://a.org/sqlite"]


def test_what_a_joiner_writes_is_limited(served):
    server, local, url = served
    joiner = RemoteBoard(url, server.key, "laptop")
    joiner.post("check", 0, not_for="agent-1", query="x" * 500)
    assert local.claim("remote-laptop").kind == "research", "it can't take the check it posted, whatever it asked for"
    joiner.set_check(0, joiner.name, "AGREE", "both say 2000\u200b " + "y" * 500)
    check = local.checks()[0]
    assert check["agent"] == "remote-laptop" and len(check["reason"]) == 200 and "\u200b" not in check["reason"]
    for bad in [
        lambda: joiner.set_check(0, joiner.name, "TRUST ME", ""),
        lambda: joiner.set_check(7, joiner.name, "AGREE", ""),
        lambda: joiner.post("research", 0, not_for="x", attempt=9),
        lambda: joiner.post("shutdown", 0, not_for="x"),
    ]:
        with pytest.raises(BoardGone):
            bad()


def test_a_task_whose_agent_vanished_is_given_to_someone_else(tmp_path):
    board = Board(tmp_path / "b.sqlite", lease_seconds=0.01)
    board.start(1)
    first = board.claim("remote-gone")
    import time

    time.sleep(0.05)
    again = board.claim("agent-1")
    assert again.id == first.id and board.held_by(first.id) == "agent-1"


async def test_a_host_and_a_joiner_finish_a_run_together(tmp_path):
    """One agent on the host and one joining over the network share the questions."""
    settings = SETTINGS.__class__(**{**vars(SETTINGS), "pages_per_question": 1, "team_agents": 1})
    board_path = tmp_path / "board.sqlite"
    servers, joiners = [], []

    def join(url, key):
        async def work():
            board = RemoteBoard(url, key, "laptop")
            researcher = Researcher(RuleModel(calls=[]), web(RESULTS, [], []), settings)
            try:
                await _work(board.name, board, researcher, board.questions(), lambda: False, settings)
            except BoardGone:
                pass  # The host finished and closed its board.

        asyncio.run(work())

    async def open_to_joiners(questions):
        server = BoardServer(
            board_path, questions, host_reads, asyncio.get_running_loop(), 900, 2, listen="127.0.0.1", port=0
        )
        servers.append(server)
        thread = threading.Thread(target=join, args=(f"http://127.0.0.1:{server.port}", server.key), daemon=True)
        thread.start()
        joiners.append(thread)
        while not server.joined:
            await asyncio.sleep(0.05)

    graph = build_research_graph(
        RuleModel(calls=[]), web(RESULTS, [], []), settings, team=team_stage(board_path, settings, open_to_joiners)
    )
    try:
        state = await graph.ainvoke(initial_research_state("Compare them"), {"recursion_limit": 100})
    finally:
        servers[0].close()
    joiners[0].join(timeout=10)

    agents = {note["agent"] for note in state["notes"]}
    assert agents == {"agent-1", "remote-laptop"}, "both took part"
    assert state["status"] == "done"
    assert {check["verdict"] for check in state["checks"]} == {"AGREE"}
    assert "found by remote-laptop" in state["report"]

"""Agents sharing a board, tested with a rule-following model and a made-up web."""

import re
import time
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent.config import Settings
from agent.research import Findings, build_research_graph, initial_research_state
from agent.team import NO_SECOND_SOURCE, Board, parse_verdict, team_stage
from agent.tools import Pacer

SETTINGS = Settings()
QUESTIONS = "When was SQLite released?\nWhen was DuckDB released?"
PAGES = {
    "https://a.org/sqlite": "SQLite was first released in August 2000.",
    "https://b.org/sqlite": "A page about something else entirely.",
    "https://c.org/sqlite": "The first SQLite release came out in the year 2000.",
    "https://a.org/duckdb": "DuckDB was first released in 2019.",
    "https://d.org/duckdb": "The initial DuckDB release dates from 2019.",
    "https://e.org/empty": "Nothing useful on this page at all.",
    "https://f.org/quokka": "Quokkas were first described in 1696.",
}
FACTS = {
    "August 2000": "SQLite was first released in August 2000.",
    "the year 2000": "The first SQLite release came out in the year 2000.",
    "released in 2019": "DuckDB was first released in 2019.",
    "dates from 2019": "The initial DuckDB release dates from 2019.",
    "1696": "Quokkas were first described in 1696.",
}


class RuleModel(BaseChatModel):
    """Answers each kind of request the way a careful model would, whatever order they arrive in."""

    questions: str = QUESTIONS
    compare: str = "AGREE - both give the same year."
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "rules"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system, content = messages[0].content, messages[1].content
        self.calls.append((system, content))
        if system.startswith("You break a research task"):
            reply = self.questions
        elif system.startswith("You pick which search results"):
            reply = "1, 2"
        elif system.startswith("You extract evidence"):
            source = content.split("Source text:\n", 1)[1]
            reply = next((f"QUOTE: {quote}\nFACT: It says {key}." for key, quote in FACTS.items() if quote in source), "NONE")
        elif system.startswith("A web search for this question found nothing"):
            reply = "quokka first described year"
        elif system.startswith("Two researchers looked"):
            reply = self.compare
        elif system.startswith("You write the final report"):
            sources = dict.fromkeys(re.findall(r"Source: (\S+)", content))
            reply = " ".join(f"A fact ({source})." for source in sources)
        else:
            reply = "PASS"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply))])


def web(results, searches, opened):
    @tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search."""
        searches.append(query)
        urls = next((urls for key, urls in results.items() if key in query), [])
        return "\n\n".join(f"{i}. Title\nURL: {url}\nSnippet" for i, url in enumerate(urls, 1))

    @tool
    async def web_read_page(url: str, offset: int = 0) -> str:
        """Read."""
        opened.append(url)
        return f"Title: T\nURL: {url}\n\n{PAGES[url]}"

    return [web_search, web_read_page]


RESULTS = {
    "SQLite": ["https://a.org/sqlite", "https://c.org/sqlite"],
    "DuckDB": ["https://a.org/duckdb", "https://d.org/duckdb"],
}


async def run(tmp_path, results=RESULTS, model=None, settings=SETTINGS, start=None):
    searches, opened = [], []
    model = model or RuleModel(calls=[])
    # One page per question, so the made-up web's second source is left for the check.
    settings = replace(settings, pages_per_question=1)
    graph = build_research_graph(
        model, web(results, searches, opened), settings, team=team_stage(tmp_path / "board.sqlite", settings)
    )
    state = await graph.ainvoke(initial_research_state("Compare them") | (start or {}), {"recursion_limit": 100})
    return state, Board(tmp_path / "board.sqlite"), searches, opened, model


async def test_each_question_is_researched_once_and_checked_by_the_other_agent(tmp_path):
    state, board, _, opened, _ = await run(tmp_path)
    assert state["status"] == "done"
    researchers = {note["question"]: note["agent"] for note in state["notes"] if "a.org" in note["source"]}
    assert set(researchers.values()) == {"agent-1", "agent-2"}, "the two questions go to different agents"
    for check in state["checks"]:
        assert check["verdict"] == "AGREE"
        assert check["agent"] != researchers[check["question"]], "nobody checks their own work"
    assert len(opened) == len(set(opened)), "no page is read twice"
    assert board.unfinished() == 0


async def test_a_check_only_counts_sources_the_first_agent_did_not_use(tmp_path):
    state, _, _, _, _ = await run(tmp_path)
    by_question = {}
    for note in state["notes"]:
        by_question.setdefault(note["question"], {}).setdefault(note["agent"], set()).add(note["source"])
    for sources_by_agent in by_question.values():
        first, second = sources_by_agent.values()
        assert not first & second
    assert {"https://c.org/sqlite", "https://d.org/duckdb"} <= {note["source"] for note in state["notes"]}


async def test_report_shows_who_found_what_and_the_cross_checks(tmp_path):
    state, _, _, _, model = await run(tmp_path)
    assert "(found by agent-" in state["report"]
    assert "## Cross-checks" in state["report"]
    assert "Sub-question 1 (When was SQLite released?): AGREE. both give the same year" in state["report"]
    writer_input = next(content for system, content in model.calls if system.startswith("You write the final report"))
    assert "A second researcher cross-checked these" in writer_input


async def test_disagreement_is_passed_on_to_the_writer_and_the_reader(tmp_path):
    model = RuleModel(calls=[], compare="DISAGREE - one says 2000 and the other 2001.")
    state, _, _, _, model = await run(tmp_path, model=model)
    assert {check["verdict"] for check in state["checks"]} == {"DISAGREE"}
    assert "DISAGREE. one says 2000 and the other 2001" in state["report"]
    writer_input = next(content for system, content in model.calls if system.startswith("You write the final report"))
    assert "DISAGREE" in writer_input


async def test_an_agent_that_finds_nothing_hands_the_question_to_the_other(tmp_path):
    results = {"When were quokkas": ["https://e.org/empty"], "quokka first described": ["https://e.org/empty", "https://f.org/quokka"]}
    model = RuleModel(calls=[], questions="When were quokkas first described?")
    state, board, searches, opened, _ = await run(tmp_path, results=results, model=model)
    log = [entry["text"] for entry in board.log()]
    assert any("handed it over reworded" in text for text in log)
    assert "quokka first described year" in searches
    assert opened.count("https://e.org/empty") == 1, "the second agent skips the page the first one read"
    note = state["notes"][0]
    assert note["source"] == "https://f.org/quokka"
    first_agent = next(entry["agent"] for entry in board.log() if "researching" in entry["text"])
    assert note["agent"] != first_agent


async def test_an_answer_with_no_second_source_is_marked_unconfirmed(tmp_path):
    results = {"DuckDB": ["https://a.org/duckdb"], "SQLite": ["https://a.org/sqlite"]}
    state, _, _, _, _ = await run(tmp_path, results=results)
    assert [check["verdict"] for check in state["checks"]] == ["UNCONFIRMED", "UNCONFIRMED"]
    assert NO_SECOND_SOURCE in state["report"]
    assert state["status"] == "done"


async def test_time_limit_stops_the_team_and_says_so(tmp_path):
    settings = replace(SETTINGS, max_minutes=1)
    state, board, _, opened, _ = await run(tmp_path, settings=settings, start={"started_at": time.time() - 3600})
    assert opened == []
    assert state["status"] == "budget_exhausted"
    assert board.unfinished() == 2


def test_a_task_can_only_be_claimed_once_and_never_by_the_excluded_agent(tmp_path):
    board = Board(tmp_path / "b.sqlite")
    board.start(1)
    board.post("check", 0, not_for="agent-1")
    first = board.claim("agent-1")
    assert first.kind == "research", "research comes before checking"
    assert board.claim("agent-1") is None, "the only task left isn't allowed for this agent"
    second = board.claim("agent-2")
    assert second.kind == "check"
    assert board.claim("agent-2") is None and board.in_progress() == 2
    board.finish(first)
    board.finish(second)
    assert board.unfinished() == 0


def test_a_resumed_run_reopens_interrupted_tasks_without_adding_more(tmp_path):
    board = Board(tmp_path / "b.sqlite")
    board.start(2)
    task = board.claim("agent-1")
    board.record("agent-2", 1, Findings(notes=[{"question": 1, "source": "s", "quote": "q", "fact": "f"}], opened=["https://x.org"]))
    board.close()

    board = Board(tmp_path / "b.sqlite")
    board.start(2)
    assert board.in_progress() == 0 and board.unfinished() == 2
    assert board.claim("agent-2").id == task.id
    assert board.notes(1)[0]["agent"] == "agent-2" and board.opened(1) == {"https://x.org"}


def test_verdicts_are_read_from_untidy_replies():
    assert parse_verdict("**AGREE** - both say 2000.\nMore text") == ("AGREE", "both say 2000")
    assert parse_verdict("disagree: the years differ") == ("DISAGREE", "the years differ")
    assert parse_verdict("I think they match") == ("UNCLEAR", "I think they match")
    assert parse_verdict("") == ("UNCLEAR", "")


async def test_searches_are_spaced_out():
    pacer = Pacer(0.2)
    started = time.monotonic()
    for _ in range(3):
        await pacer.wait()
    assert 0.4 <= time.monotonic() - started < 1.0, "the first goes at once, the next two wait"

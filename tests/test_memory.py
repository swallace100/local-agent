"""Remembering what earlier runs found, tested with embeddings built from word counts."""

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from test_graph import ScriptedModel

from agent.config import Settings
from agent.memory import Memory
from agent.research import build_research_graph, initial_research_state

SETTINGS = Settings()
QUESTIONS = ["When was SQLite first released?", "What license does DuckDB use?"]
NOTES = [
    {"question": 0, "source": "https://a.org/sqlite", "quote": "SQLite was first released in August 2000.", "fact": "SQLite came out in 2000."},
    {"question": 1, "source": "https://d.org/duckdb", "quote": "DuckDB is released under the MIT License.", "fact": "DuckDB uses the MIT license."},
    {"question": 1, "source": "knowledge/private.md", "quote": "Our DuckDB cluster is in rack 4.", "fact": "A private note."},
]


class WordEmbeddings(Embeddings):
    """Texts that share words point the same way, which is enough to tell related from unrelated."""

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 512
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 512] += 1
        length = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / length for value in vector]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def memory(tmp_path):
    return Memory(SETTINGS, store_dir=tmp_path / "chroma", embeddings=WordEmbeddings())


def test_only_notes_from_web_pages_are_remembered(memory):
    assert memory.remember("run-1", "Compare the databases", QUESTIONS, NOTES) == 2
    sources = {doc.metadata["source"] for doc, _ in memory.recall("DuckDB license cluster rack", limit=10)}
    assert "knowledge/private.md" not in sources


def test_a_note_found_again_replaces_the_earlier_copy(memory):
    memory.remember("run-1", "task", QUESTIONS, NOTES)
    memory.remember("run-2", "task", QUESTIONS, NOTES[:1] * 2)
    found = memory.recall("When was SQLite first released?", limit=10)
    assert [doc.metadata["run"] for doc, _ in found if "sqlite" in doc.metadata["source"]] == ["run-2"]


def test_leads_are_pages_that_answered_a_similar_question(memory, monkeypatch):
    # The real limit is tuned to the real embedding model. Word counts are much cruder.
    monkeypatch.setattr("agent.memory.LEAD_DISTANCE", 0.6)
    memory.remember("run-1", "task", QUESTIONS, NOTES)
    assert memory.leads("When was SQLite first released?", limit=2) == ["https://a.org/sqlite"]
    assert memory.leads("Tallest mountain Peru", limit=2) == []


def test_recall_shows_the_fact_its_quote_and_where_it_came_from(memory):
    memory.remember("run-1", "Compare the databases", QUESTIONS, NOTES)
    document, _ = memory.recall("DuckDB MIT license")[0]
    assert document.page_content.split("\n") == [QUESTIONS[1], "DuckDB uses the MIT license.", "Quote: DuckDB is released under the MIT License."]
    assert document.metadata["source"] == "https://d.org/duckdb" and document.metadata["task"] == "Compare the databases"


def web(searches, opened, pages):
    @tool
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search."""
        searches.append(query)
        return "1. Title\nURL: https://new.org/found\nSnippet"

    @tool
    async def web_read_page(url: str, offset: int = 0) -> str:
        """Read."""
        opened.append(url)
        return f"Title: T\nURL: {url}\n\n{pages[url]}"

    return [web_search, web_read_page]


async def research(replies, leads, pages):
    searches, opened = [], []
    model = ScriptedModel(replies=[AIMessage(r) for r in replies], calls=[])
    graph = build_research_graph(model, web(searches, opened, pages), SETTINGS, leads=leads)
    state = await graph.ainvoke(initial_research_state("When was SQLite released?"))
    return state, searches, opened, model


async def test_a_remembered_page_is_read_again_instead_of_searching():
    pages = {"https://a.org/sqlite": "SQLite was first released in August 2000."}
    note = "QUOTE: SQLite was first released in August 2000.\nFACT: It came out in 2000."
    state, searches, opened, _ = await research(
        ["When was SQLite released?", note, "In 2000 (https://a.org/sqlite).", "PASS"],
        leads=lambda question: ["https://a.org/sqlite"],
        pages=pages,
    )
    assert searches == [], "the remembered page answered it, so no search was made"
    assert opened == ["https://a.org/sqlite"]
    assert state["status"] == "done"
    assert state["notes"][0]["quote"] == "SQLite was first released in August 2000.", "checked against the page as it is now"


async def test_a_remembered_page_that_no_longer_helps_falls_back_to_searching():
    pages = {
        "https://a.org/sqlite": "This page has moved.",
        "https://new.org/found": "SQLite was first released in August 2000.",
    }
    note = "QUOTE: SQLite was first released in August 2000.\nFACT: It came out in 2000."
    state, searches, opened, model = await research(
        ["When was SQLite released?", "NONE", note, "In 2000 (https://new.org/found).", "PASS"],
        leads=lambda question: ["https://a.org/sqlite"],
        pages=pages,
    )
    assert searches == ["When was SQLite released?"]
    assert opened == ["https://a.org/sqlite", "https://new.org/found"]
    assert state["notes"][0]["source"] == "https://new.org/found"
    assert not any("SQLite came out" in str(message.content) for messages in model.calls for message in messages), (
        "what an earlier run wrote down never reaches the model"
    )

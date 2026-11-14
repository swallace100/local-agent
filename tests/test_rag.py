"""Reading documents and searching them, tested with made-up embeddings so no model is needed."""

import hashlib

import pytest
from langchain_core.embeddings import Embeddings
from make_docs import CONTRACT_PAGES, ONBOARDING_PARAGRAPHS, make_docx, make_pdf

from agent.config import Settings
from agent.rag import Knowledge, fuse, keyword_rank, read_parts
from agent.tools import knowledge_tool


class UnrelatedEmbeddings(Embeddings):
    """Gives every text its own direction, so nothing is ever close in meaning to a query."""

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [byte - 127.5 for byte in digest]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def knowledge(tmp_path):
    folder = tmp_path / "knowledge"
    folder.mkdir()
    (folder / "contract.pdf").write_bytes(make_pdf(CONTRACT_PAGES))
    (folder / "onboarding.docx").write_bytes(make_docx(ONBOARDING_PARAGRAPHS))
    (folder / "bread.md").write_text("Sourdough needs a hydration of about 75 percent.")
    (folder / "scan.pdf").write_bytes(make_pdf([[]]))
    (folder / "broken.docx").write_bytes(b"this is not a zip file")
    return Knowledge(Settings(), folder=folder, store_dir=tmp_path / "chroma", embeddings=UnrelatedEmbeddings())


def test_pdf_text_is_read_page_by_page(tmp_path):
    path = tmp_path / "c.pdf"
    path.write_bytes(make_pdf(CONTRACT_PAGES))
    parts = read_parts(path)
    assert [page for page, _ in parts] == [1, 2]
    assert "renews on 14 March" in parts[0][1] and "Ilse Brandt" in parts[1][1]


def test_word_paragraphs_are_read(tmp_path):
    path = tmp_path / "o.docx"
    path.write_bytes(make_docx(["First & foremost", "", "Laptop model TX-880"]))
    assert read_parts(path) == [(0, "First & foremost\n\nLaptop model TX-880")]


def test_unreadable_documents_give_a_reason(tmp_path):
    scan = tmp_path / "scan.pdf"
    scan.write_bytes(make_pdf([[]]))
    with pytest.raises(ValueError, match="probably a scan"):
        read_parts(scan)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf at all")
    with pytest.raises(ValueError, match="couldn't be read"):
        read_parts(broken)


def test_index_reports_what_it_skipped(knowledge):
    result = knowledge.index()
    assert result.added == 3
    assert sorted(reason.split(":")[0] for reason in result.skipped) == ["broken.docx", "scan.pdf"]
    again = knowledge.index()
    assert again.unchanged == 3 and again.added == 0


def test_exact_words_are_found_when_meaning_search_finds_nothing(knowledge):
    knowledge.index()
    results = knowledge.search("TX-880")
    assert [doc.metadata["source"] for doc, _ in results] == ["onboarding.docx"]
    assert results[0][1] is None, "found by its words, so there is no distance"
    document, _ = knowledge.search("Who is the vendor contact for Kestrel?")[0]
    assert document.metadata["source"] == "contract.pdf" and document.metadata["page"] == 2


def test_unrelated_queries_still_find_nothing(knowledge):
    knowledge.index()
    assert knowledge.search("What is the capital of Mongolia?") == []
    assert knowledge.search("the of and") == []


async def test_search_results_name_the_page(knowledge):
    knowledge.index()
    output = await knowledge_tool(knowledge).ainvoke({"query": "Kestrel vendor contact"})
    assert output.startswith("Source: knowledge/contract.pdf, page 2\n")
    assert "Source: knowledge/bread.md\n" in await knowledge_tool(knowledge).ainvoke({"query": "sourdough hydration"})


def test_keyword_rank_needs_most_of_the_query_words():
    texts = [
        "The Heron service listens on port 7421.",
        "Port wine pairs well with cheese.",
        "Escalation code ESC-4417 is for outages. ESC-4417 goes to the duty manager.",
        "Escalation codes are listed in the handbook.",
    ]
    assert keyword_rank("Heron port", texts, 4) == [0], "one shared word isn't enough"
    assert keyword_rank("ESC-4417", texts, 4) == [2]
    assert keyword_rank("escalation code handbook", texts, 4) == [3, 2], "plurals match, and all three words beat two"
    assert keyword_rank("what is the", texts, 4) == [], "only common words"
    assert keyword_rank("heron", [], 4) == []


def test_fusion_prefers_what_both_rankings_agree_on():
    assert fuse([["a", "b", "c"], ["c", "b", "d"]], 3)[0] in ("b", "c")
    assert fuse([["a", "b", "c"], ["b", "x", "y"]], 2) == ["b", "a"]
    assert fuse([[], ["k"]], 4) == ["k"]

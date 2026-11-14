"""RAG over a local folder: split files into chunks, embed them, and search them.

Search combines two rankings. Embeddings find text that means the same as the query, and
keyword matching finds exact words that embeddings blur, such as codes, names and numbers.
"""

import hashlib
import math
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from agent.config import Settings

TEXT_SUFFIXES = {
    ".md", ".txt", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".html", ".csv",
}
DOCUMENT_SUFFIXES = {".pdf", ".docx"}
CHUNK_CHARS = 1_000
CHUNK_OVERLAP = 150
# Chunks further away than this (cosine distance) are treated as unrelated.
MAX_DISTANCE = 0.45

# A chunk only counts as a keyword match if it has this share of the query's words, so a
# single common word can't pull in something unrelated.
MIN_TERM_SHARE = 0.6
# The standard BM25 settings: how fast repeated words stop adding, and how much length matters.
BM25_K1, BM25_B = 1.5, 0.75
# The usual constant for combining rankings. It keeps one first place from outweighing
# a good position in both.
RANK_FUSION_K = 60
WORD = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a about according an and any are as at be by can could did do does for from has have how i in is it "
    "its me my notes of on or our say says should tell that the their them there these they this to was "
    "we were what when where which who whom why will with would you your".split()
)
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class PrefixedEmbeddings(OllamaEmbeddings):
    """nomic-embed-text expects a prefix saying whether text is a document or a query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


@dataclass
class IndexResult:
    added: int
    removed: int
    unchanged: int
    chunks: int
    # Files that couldn't be indexed, each with the reason.
    skipped: list[str] = field(default_factory=list)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_files(folder: Path) -> dict[str, Path]:
    files = {}
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        hidden = any(part.startswith(".") for part in relative.parts)
        if path.is_file() and not hidden and path.suffix.lower() in TEXT_SUFFIXES | DOCUMENT_SUFFIXES:
            files[relative.as_posix()] = path
    return files


def _read_docx(path: Path) -> str:
    """Read a Word file's paragraphs. A .docx is a zip with the text in one XML file."""
    with zipfile.ZipFile(path) as archive:
        body = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = (
        "".join(node.text or "" for node in paragraph.iter(f"{WORD_NAMESPACE}t"))
        for paragraph in body.iter(f"{WORD_NAMESPACE}p")
    )
    return "\n\n".join(text for text in paragraphs if text.strip())


def read_parts(path: Path) -> list[tuple[int, str]]:
    """Return a file's text as (page, text) parts. The page is 0 for files without pages.

    Raises ValueError with a reason a person can act on if the file has no usable text.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise ValueError("it is password protected")
            parts = [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
            if not any(text.strip() for _, text in parts):
                raise ValueError("it has no text in it, so it is probably a scan")
            return parts
        if suffix == ".docx":
            return [(0, _read_docx(path))]
    except ValueError:
        raise
    except Exception as error:  # Document parsers raise many types on damaged files.
        raise ValueError(f"it couldn't be read ({type(error).__name__})") from error
    return [(0, path.read_text(encoding="utf-8", errors="replace"))]


def _terms(text: str) -> list[str]:
    """The words that matter for matching, with plain plurals folded into the singular."""
    words = (word for word in WORD.findall(text.lower()) if word not in STOPWORDS)
    return [word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word for word in words]


def keyword_rank(query: str, texts: Sequence[str], limit: int) -> list[int]:
    """Return the positions of the texts that best match the query's words, best first (BM25)."""
    wanted = set(_terms(query))
    if not wanted or not texts:
        return []
    counts = [Counter(_terms(text)) for text in texts]
    average_length = sum(sum(count.values()) for count in counts) / len(counts) or 1
    containing = {term: sum(1 for count in counts if term in count) for term in wanted}
    needed = math.ceil(len(wanted) * MIN_TERM_SHARE)

    scored = []
    for position, count in enumerate(counts):
        matched = [term for term in wanted if term in count]
        if len(matched) < needed:
            continue
        length = sum(count.values())
        score = 0.0
        for term in matched:
            rarity = math.log(1 + (len(counts) - containing[term] + 0.5) / (containing[term] + 0.5))
            frequency = count[term]
            score += rarity * frequency * (BM25_K1 + 1) / (
                frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / average_length)
            )
        scored.append((score, position))
    return [position for _, position in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def fuse(rankings: Sequence[Sequence[str]], limit: int) -> list[str]:
    """Merge rankings of the same items, favouring whatever ranks well in more than one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1 / (RANK_FUSION_K + rank + 1)
    return sorted(scores, key=lambda key: -scores[key])[:limit]


def _key(document: Document) -> str:
    return f"{document.metadata['source']}\n{document.page_content}"


class Knowledge:
    def __init__(
        self,
        settings: Settings,
        folder: Path | None = None,
        store_dir: Path | None = None,
        embeddings: Embeddings | None = None,
        collection: str = "knowledge",
    ):
        self.folder = folder or settings.knowledge_dir
        store_dir = store_dir or settings.vector_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        # num_gpu=0 keeps embeddings on the CPU so they don't push the main model off the GPU.
        embeddings = embeddings or PrefixedEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_url, num_gpu=0
        )
        self.store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(store_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._chunks: list[Document] | None = None

    def _indexed(self) -> dict[str, dict]:
        """Return each indexed source with its file hash and chunk IDs."""
        data = self.store.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for chunk_id, metadata in zip(data["ids"], data["metadatas"]):
            entry = sources.setdefault(metadata["source"], {"hash": metadata["hash"], "ids": []})
            entry["ids"].append(chunk_id)
        return sources

    def index(self) -> IndexResult:
        """Bring the index in line with the folder, only re-embedding changed files."""
        files = _find_files(self.folder) if self.folder.exists() else {}
        hashes = {source: _file_hash(path) for source, path in files.items()}
        indexed = self._indexed()

        stale = [s for s, entry in indexed.items() if hashes.get(s) != entry["hash"]]
        for source in stale:
            self.store.delete(ids=indexed[source]["ids"])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP
        )
        changed = [s for s in files if s not in indexed or s in stale]
        skipped = []
        for source in changed:
            try:
                parts = read_parts(files[source])
            except ValueError as reason:
                skipped.append(f"{source}: {reason}")
                continue
            documents = [
                Document(page_content=chunk, metadata={"source": source, "hash": hashes[source], "page": page})
                for page, text in parts
                for chunk in splitter.split_text(text)
            ]
            if documents:
                self.store.add_documents(documents, ids=[f"{source}#{i}" for i in range(len(documents))])

        self._chunks = None
        total = len(self.store.get(include=[])["ids"])
        return IndexResult(
            added=len(changed) - len(skipped),
            removed=len(set(stale) - set(files)),
            unchanged=len(files) - len(changed),
            chunks=total,
            skipped=skipped,
        )

    def add(self, documents: list[Document], ids: list[str]) -> None:
        """Store documents directly, replacing any that have the same ID."""
        if documents:
            self.store.add_documents(documents, ids=ids)
            self._chunks = None

    def _all_chunks(self) -> list[Document]:
        """Every chunk, loaded once for keyword search. A personal notes folder is small enough."""
        if self._chunks is None:
            data = self.store.get(include=["documents", "metadatas"])
            self._chunks = [
                Document(page_content=text, metadata=metadata)
                for text, metadata in zip(data["documents"], data["metadatas"])
            ]
        return self._chunks

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float | None]]:
        """Return the best chunks as (document, distance), dropping unrelated ones.

        The distance is None for a chunk found only by its words.
        """
        chunks = self._all_chunks()
        if not chunks:
            return []
        by_meaning = [
            (document, distance)
            for document, distance in self.store.similarity_search_with_score(query, k=k)
            if distance <= MAX_DISTANCE
        ]
        by_words = [chunks[position] for position in keyword_rank(query, [c.page_content for c in chunks], k)]

        found: dict[str, tuple[Document, float | None]] = {_key(document): (document, None) for document in by_words}
        found |= {_key(document): (document, distance) for document, distance in by_meaning}
        order = fuse([[_key(document) for document, _ in by_meaning], [_key(document) for document in by_words]], k)
        return [found[key] for key in order]

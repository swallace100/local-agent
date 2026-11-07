"""RAG over a local folder: split files into chunks, embed them, and search them."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import Settings

TEXT_SUFFIXES = {
    ".md", ".txt", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".html", ".csv",
}
CHUNK_CHARS = 1_000
CHUNK_OVERLAP = 150
# Chunks further away than this (cosine distance) are treated as unrelated.
MAX_DISTANCE = 0.45


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


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_files(folder: Path) -> dict[str, Path]:
    files = {}
    for path in folder.rglob("*"):
        relative = path.relative_to(folder)
        hidden = any(part.startswith(".") for part in relative.parts)
        if path.is_file() and not hidden and path.suffix.lower() in TEXT_SUFFIXES:
            files[relative.as_posix()] = path
    return files


class Knowledge:
    def __init__(self, settings: Settings, folder: Path | None = None, store_dir: Path | None = None):
        self.folder = folder or settings.knowledge_dir
        store_dir = store_dir or settings.vector_dir
        store_dir.mkdir(parents=True, exist_ok=True)
        # num_gpu=0 keeps embeddings on the CPU so they don't push the main model off the GPU.
        embeddings = PrefixedEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_url, num_gpu=0
        )
        self.store = Chroma(
            collection_name="knowledge",
            embedding_function=embeddings,
            persist_directory=str(store_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

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
        for source in changed:
            text = files[source].read_text(encoding="utf-8", errors="replace")
            chunks = splitter.split_text(text)
            if not chunks:
                continue
            self.store.add_documents(
                [
                    Document(page_content=c, metadata={"source": source, "hash": hashes[source]})
                    for c in chunks
                ],
                ids=[f"{source}#{i}" for i in range(len(chunks))],
            )

        total = len(self.store.get(include=[])["ids"])
        return IndexResult(
            added=len(changed),
            removed=len(set(stale) - set(files)),
            unchanged=len(files) - len(changed),
            chunks=total,
        )

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return the closest chunks as (document, distance), dropping unrelated ones."""
        if not self.store.get(limit=1, include=[])["ids"]:
            return []
        results = self.store.similarity_search_with_score(query, k=k)
        return [(doc, distance) for doc, distance in results if distance <= MAX_DISTANCE]

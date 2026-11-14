"""What earlier research runs found, kept so that later runs can start from it.

Every checked note from a research or team run is saved here with the page it came from.
It is kept apart from your own notes in knowledge/, and trusted less, because it began as
text on a web page that passed through the model.

For that reason a later run never uses a remembered note as evidence. It uses it as a lead:
the page that answered a similar question before is read again, and its quotes are checked
afresh. That also saves a web search, which matters when search engines limit how often
you can ask.
"""

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agent.config import Settings
from agent.rag import Knowledge

# A remembered note this close in meaning to the question (cosine distance) is worth
# revisiting. Measured with nomic-embed-text: questions about the same subject came within
# 0.11 of its notes, and the same question about a different subject ("Who started Go?"
# against notes on Rust) was 0.21 or more away.
LEAD_DISTANCE = 0.15
LEAD_CANDIDATES = 8
TASK_CHARS = 200


class Memory:
    def __init__(self, settings: Settings, store_dir: Path | None = None, embeddings: Embeddings | None = None):
        self._store = Knowledge(settings, store_dir=store_dir, embeddings=embeddings, collection="findings")

    def remember(self, run_id: str, task: str, questions: Sequence[str], notes: Sequence[dict]) -> int:
        """Save a run's notes that came from web pages. Returns how many were saved.

        Notes from your own files are left out, since those files are already searchable.
        A note found again replaces the earlier copy, so the date shows when it was last seen.
        """
        documents, ids = [], []
        for note in notes:
            if not note["source"].startswith(("http://", "https://")):
                continue
            question = questions[note["question"]]
            documents.append(
                Document(
                    page_content=f"{question}\n{note['fact']}\nQuote: {note['quote']}",
                    metadata={
                        "source": note["source"],
                        "run": run_id,
                        "date": date.today().isoformat(),
                        "task": task[:TASK_CHARS],
                    },
                )
            )
            ids.append(hashlib.sha256(f"{note['source']}\n{note['quote']}".encode()).hexdigest())
        # The same quote can be noted twice in one run, and a batch can't repeat an ID.
        unique = dict(zip(ids, documents))
        self._store.add(list(unique.values()), list(unique))
        return len(unique)

    def recall(self, query: str, limit: int = 6) -> list[tuple[Document, float | None]]:
        """Return remembered notes that match the query, by meaning or by exact words."""
        return self._store.search(query, k=limit)

    def leads(self, question: str, limit: int) -> list[str]:
        """Return pages that answered a similar question before, best first."""
        urls = []
        for document, distance in self._store.search(question, k=LEAD_CANDIDATES):
            # A match on words alone isn't enough: questions about different things share words.
            close = distance is not None and distance <= LEAD_DISTANCE
            if close and document.metadata["source"] not in urls:
                urls.append(document.metadata["source"])
        return urls[:limit]

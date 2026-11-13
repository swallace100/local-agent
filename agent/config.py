"""Settings for the agent. Any value can be overridden with an AGENT_* environment variable."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"AGENT_{name}", default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://127.0.0.1:11434"))
    searxng_url: str = field(default_factory=lambda: _env("SEARXNG_URL", "http://127.0.0.1:8888"))
    model: str = field(default_factory=lambda: _env("MODEL", "agent"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text"))
    sandbox_image: str = field(default_factory=lambda: _env("SANDBOX_IMAGE", "python:3.13-slim"))

    knowledge_dir: Path = field(default_factory=lambda: Path(_env("KNOWLEDGE_DIR", str(ROOT / "knowledge"))))
    runs_dir: Path = field(default_factory=lambda: Path(_env("RUNS_DIR", str(ROOT / "runs"))))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / ".data"))))

    # Keep below num_ctx in ollama/Modelfile, leaving room for the model's reply.
    context_tokens: int = 24_000
    max_steps: int = 30
    max_minutes: int = 45
    max_revisions: int = 2
    # Each section is about 8,000 characters.
    max_sections_per_page: int = 3
    command_timeout_seconds: int = 120

    # Research runs (agent research). Their steps are small model calls, so more are allowed.
    research_max_steps: int = 60
    max_questions: int = 5
    search_results: int = 6
    pages_per_question: int = 2
    research_sections_per_page: int = 2
    facts_per_source: int = 4

    # Organize runs (agent organize).
    organize_max_files: int = 2_000
    organize_batch: int = 25
    organize_max_folders: int = 12

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "chroma"

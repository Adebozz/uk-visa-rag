"""Typed access to configs/config.yaml and configs/corpus.yaml."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/config.yaml")
DEFAULT_CORPUS = Path("configs/corpus.yaml")


@dataclass
class IngestionConfig:
    api_base: str = "https://www.gov.uk/api/content"
    raw_dir: str = "data/raw"
    chunks_path: str = "data/chunks.parquet"
    max_chunk_chars: int = 1400
    overlap_chars: int = 200


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    embeddings_path: str = "data/embeddings.npy"


@dataclass
class RetrievalConfig:
    top_k: int = 6
    candidates_k: int = 20
    rrf_k: int = 60


@dataclass
class LlmConfig:
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 1024
    min_retrieval_score: float = 0.02


@dataclass
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LlmConfig(**raw.get("llm", {})),
    )


def load_corpus(path: Path | str = DEFAULT_CORPUS) -> list[str]:
    return yaml.safe_load(Path(path).read_text())["pages"]

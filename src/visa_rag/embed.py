"""Build the dense index: embed every chunk with a local model.

    python -m visa_rag.embed

Embeddings are free and local (sentence-transformers); no API cost.
BGE models expect a query prefix at search time - see retrieval.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from visa_rag.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def embed_corpus(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    from sentence_transformers import SentenceTransformer  # heavy import

    df = pd.read_parquet(cfg.ingestion.chunks_path)
    model = SentenceTransformer(cfg.embedding.model_name)
    # Embed provenance + text so "how much does a student visa cost" can match
    # a chunk whose body says only "The fee is £524".
    texts = (df["doc_title"] + " - " + df["section"] + "\n" + df["text"]).tolist()
    logger.info("Embedding %d chunks with %s ...", len(texts), cfg.embedding.model_name)
    vectors = model.encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    out = Path(cfg.embedding.embeddings_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors.astype(np.float32))
    logger.info("Saved %s (%s)", out, vectors.shape)


if __name__ == "__main__":
    embed_corpus()

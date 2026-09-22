"""
Thermal scenario data store — retrieval over the deep-space thermal dataset.

Provides the knowledge layer that the ThermalAgent queries instead of relying
on a fine-tuned model. Scenarios from the HuggingFace dataset are indexed with
TF-IDF vectors and retrieved by cosine similarity.

The store is intentionally backend-agnostic: the same ``query`` interface can
be backed by a managed vector store (e.g. Amazon Bedrock Knowledge Bases) in
production by swapping the implementation, without changing the agent or tools.

Author: A Taylor
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Columns combined into the searchable text for each scenario.
TEXT_COLUMNS = ["instrument", "material_name", "environment_location", "thermal_effect"]
# Columns surfaced back to the agent in retrieval results.
RESULT_COLUMNS = TEXT_COLUMNS + ["strategy_type", "strategy_recommendation"]


class ThermalDataStore:
    """Retrievable store of thermal mitigation scenarios.

    Indexes scenario records with TF-IDF and returns the most similar prior
    scenarios for a free-text query. Acts as the agent's knowledge base.

    Author: A Taylor
    """

    def __init__(self):
        """Initialize an empty data store."""
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.records = None  # list[dict] of scenario metadata

    @staticmethod
    def _texts_from_df(df):
        """Build the searchable text for every row with vectorized string ops.

        Args:
            df: DataFrame containing some or all of the TEXT_COLUMNS.

        Returns:
            List of descriptive strings, one per row.
        """
        combined = None
        for col in TEXT_COLUMNS:
            if col not in df.columns:
                continue
            part = df[col].fillna("").astype(str).replace("nan", "", regex=False)
            combined = part if combined is None else combined + " " + part
        if combined is None:
            return [""] * len(df)
        # Collapse the extra whitespace left by empty fields.
        return combined.str.split().str.join(" ").tolist()

    @staticmethod
    def _top_k_indices(scores, top_k):
        """Return indices of the top_k scores in descending order.

        Uses argpartition (O(n)) instead of a full argsort (O(n log n)) —
        it only sorts the k winners, not the whole corpus.
        """
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(scores, -top_k)[-top_k:]
        return idx[np.argsort(scores[idx])[::-1]]

    def build(self, df):
        """Build the TF-IDF index from a DataFrame of scenarios.

        Args:
            df: DataFrame containing at least the TEXT_COLUMNS.

        Returns:
            self, to allow chaining.
        """
        texts = self._texts_from_df(df)
        self.matrix = self.vectorizer.fit_transform(texts)

        available = [c for c in RESULT_COLUMNS if c in df.columns]
        self.records = df[available].to_dict(orient="records")
        logger.info("Built data store index over %d scenarios", len(self.records))
        return self

    def query(self, text, top_k=3):
        """Retrieve the most similar scenarios for a free-text query.

        Args:
            text: Free-text description of the scenario to match.
            top_k: Number of scenarios to return.

        Returns:
            List of dicts, each a scenario record plus a 'similarity' score,
            sorted by descending similarity.

        Raises:
            RuntimeError: If the store has not been built or loaded.
        """
        if self.matrix is None or self.records is None:
            raise RuntimeError("Data store is empty. Call build() or load() first.")

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.matrix)[0]

        results = []
        for idx in self._top_k_indices(scores, top_k):
            record = dict(self.records[idx])
            record["similarity"] = float(scores[idx])
            results.append(record)
        return results

    def query_batch(self, texts, top_k=3, chunk_size=256):
        """Retrieve similar scenarios for many queries in vectorized chunks.

        Each chunk is a single sparse matrix product against the index —
        orders of magnitude faster than repeated query() calls when building
        SFT traces. chunk_size bounds the dense score block held in memory.

        Args:
            texts: List of free-text scenario descriptions.
            top_k: Number of scenarios to return per query.
            chunk_size: Number of queries scored per chunk.

        Returns:
            List with one entry per query, each a list of scenario dicts with
            'similarity' scores sorted by descending similarity.

        Raises:
            RuntimeError: If the store has not been built or loaded.
        """
        if self.matrix is None or self.records is None:
            raise RuntimeError("Data store is empty. Call build() or load() first.")

        results = []
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start:start + chunk_size]
            query_mat = self.vectorizer.transform(chunk)
            score_block = cosine_similarity(query_mat, self.matrix)
            for scores in score_block:
                hits = []
                for idx in self._top_k_indices(scores, top_k):
                    record = dict(self.records[idx])
                    record["similarity"] = float(scores[idx])
                    hits.append(record)
                results.append(hits)
        return results

    def save(self, path):
        """Persist the built index to disk.

        Args:
            path: File path for the saved index.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "records": self.records},
            path,
        )
        logger.info("Data store saved to %s", path)

    def load(self, path):
        """Load a previously built index from disk.

        Args:
            path: File path of the saved index.

        Returns:
            self, to allow chaining.
        """
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.records = data["records"]
        logger.info("Data store loaded from %s (%d scenarios)", path, len(self.records))
        return self

    @classmethod
    def from_huggingface(cls, dataset_name=None, split="train"):
        """Build a data store directly from a HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier. Defaults to the
                HF_DATASET env var or the project default.
            split: Dataset split to load.

        Returns:
            A built ThermalDataStore.
        """
        from datasets import load_dataset  # optional/heavy dependency

        name = dataset_name or os.getenv(
            "HF_DATASET", "Taylor658/deep-space-optical-chip-thermal-dataset"
        )
        logger.info("Loading dataset from HuggingFace: %s", name)
        df = load_dataset(name, split=split).to_pandas()
        return cls().build(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the thermal scenario data store. Author: A Taylor"
    )
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset id")
    parser.add_argument("--save_path", type=str, default="results/thermal_datastore.pkl")
    args = parser.parse_args()

    store = ThermalDataStore.from_huggingface(args.dataset)
    store.save(args.save_path)
    logger.info("Done — data store saved to %s", args.save_path)

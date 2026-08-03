import json
import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def default_tokenizer(text: str) -> List[str]:
    return re.findall(r'\w+', text.lower())

class BM25Index:

    def __init__(self, chunks: List[Dict[str, Any]], tokenizer: Optional[callable] = None,):

        self.tokenizer = tokenizer if tokenizer else default_tokenizer
        self.chunks = chunks

        logger.info(f"Tokenising {len(chunks)} documents")
        self.corpus_texts = [chunk["text"] for chunk in chunks]
        self.tokenized_corpus = [self.tokenizer(text) for text in self.corpus_texts]

        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.chunks)} documents.")

    def query(self, query_str: str, top_k: int = 10) -> List[Dict[str, Any]]:

        tokenized_query = self.tokenizer(query_str)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True )[:top_k]

        results = []
        for idx in top_indices:
            results.append({"chunk_id": self.chunks[idx]["chunk_id"], "score": float(scores[idx]), "text": self.chunks[idx]["text"], "metadata": self.chunks[idx]["metadata"], })
        return results

    def save(self, filepath: Path) -> None:

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {filepath}")

    @staticmethod
    def load(filepath: Path) -> "BM25Index":

        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BM25Index):
            raise TypeError(f"Loaded object is not a BM25Index. Got {type(obj)}")
        logger.info(f"BM25 index loaded from {filepath} with {len(obj.chunks)} documents.")

        return obj

    @classmethod
    def build_from_file( cls, chunks_filepath: Path, tokenizer: Optional[callable] = None,) -> "BM25Index":

        chunks = []
        with open(chunks_filepath, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_filepath}")

        return cls(chunks, tokenizer)
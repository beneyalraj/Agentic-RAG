import logging
from typing import List, Dict, Any, Optional, Callable

from src.retrieval.vector_store import PineconeVectorStore
from src.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:

    def __init__(self, dense_retriever: PineconeVectorStore, sparse_retriever: BM25Index, embed_query: Callable[[str], List[float]], rrf_k: int = 60,):

        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.embed_query = embed_query
        self.rrf_k = rrf_k

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def hybrid_search(self, query: str, top_k: int = 5, dense_candidates: int = 20, sparse_candidates: int = 20, dense_filter: Optional[Dict] = None, ) -> List[Dict[str, Any]]:

        query_embedding = self.embed_query(query)

        dense_results = self.dense.query(query_embedding=query_embedding, top_k=dense_candidates, include_metadata=True, filter=dense_filter,)

        sparse_results = self.sparse.query(query, top_k=sparse_candidates)

        dense_ranks = {res["id"]: idx + 1 for idx, res in enumerate(dense_results)}
        sparse_ranks = {res["chunk_id"]: idx + 1 for idx, res in enumerate(sparse_results)}

        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        dense_id_to_meta = {res["id"]: res for res in dense_results}
        sparse_id_to_meta = {res["chunk_id"]: res for res in sparse_results}

        combined = {}

        for cid in all_ids:
            rrf_score = 0.0
            if cid in dense_ranks:
                rrf_score += self._rrf_score(dense_ranks[cid])
            if cid in sparse_ranks:
                rrf_score += self._rrf_score(sparse_ranks[cid])

            if cid in dense_id_to_meta:
                meta = dense_id_to_meta[cid]["metadata"]
                text = meta.get("text", "")
            elif cid in sparse_id_to_meta:
                text = sparse_id_to_meta[cid]["text"]
                meta = sparse_id_to_meta[cid]["metadata"]
            else:
                continue

            combined[cid] = {"chunk_id": cid, "text": text, "metadata": meta, "rrf_score": rrf_score, "dense_rank": dense_ranks.get(cid), "sparse_rank": sparse_ranks.get(cid),}

        sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True, )

        return sorted_results[:top_k]
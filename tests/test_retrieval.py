import logging
from pathlib import Path
from config.config import settings
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.vector_store import PineconeVectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading Embedding Model...")
    embedder = get_embedding_model()

    logger.info("Connecting to Pinecone...")
    dense_store = PineconeVectorStore(api_key=settings.pinecone_api_key, index_name=settings.pinecone_index_name, cloud="aws", region="us-east-1", dimension=384,)
    dense_store.create_index_if_not_exists()

    bm25_path = Path("data/bm25_index.pkl")
    if not bm25_path.exists():
        logger.error("data/bm25_index.pkl not found! Please run scripts/build_bm25.py first.")
        return

    logger.info("Loading BM25 Index...")
    sparse_store = BM25Index.load(bm25_path)

    retriever = HybridRetriever(dense_retriever=dense_store, sparse_retriever=sparse_store, embed_query=lambda q: embedder.encode_single(q).tolist(),)

    queries = [
        "What was JP Morgan net income in recent quarter?",
        "risk factors related to credit and liquidity",
    ]

    for q in queries:
        logger.info(f"QUERY: {q}")

        results = retriever.hybrid_search(query=q, top_k=3)

        for idx, res in enumerate(results, 1):
            print(f"\n[Result {idx}] RRF Score: {res['rrf_score']:.4f} " f"(Dense Rank: {res['dense_rank']}, Sparse Rank: {res['sparse_rank']})")
            print(f"Source: {res['metadata'].get('source', 'Unknown')}")
            print(f"Text Snippet: {res['text'][:200]}...")

if __name__ == "__main__":
    main()
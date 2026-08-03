import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from config.config import settings
from src.retrieval.vector_store import PineconeVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_embedded_chunks(filepath: Path) -> List[Dict[str, Any]]:

    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} embedded chunks from {filepath}")

    return chunks


def prepare_vectors(chunks: List[Dict]) -> List[Tuple[str, List[float], Dict]]:

    vectors = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        embedding = chunk["embedding"]
        metadata = chunk["metadata"]
        vectors.append((chunk_id, embedding, metadata))

    return vectors


def main():

    input_path = Path("data/embedded_chunks.jsonl")
    if not input_path.exists():
        logger.error(f"Embedded chunks file not found: {input_path}")
        return

    vs = PineconeVectorStore(api_key=settings.pinecone_api_key, index_name=settings.pinecone_index_name, cloud=settings.pinecone_cloud, region=settings.pinecone_region, dimension=384, metric="cosine",)
    vs.create_index_if_not_exists()

    chunks = load_embedded_chunks(input_path)
    if not chunks:
        logger.error("No chunks to upsert.")
        return

    vectors = prepare_vectors(chunks)

    logger.info(f"Upserting {len(vectors)} vectors to '{settings.pinecone_index_name}'")
    batch_size = 100

    with tqdm(total=len(vectors), desc="Upserting") as pbar:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            vs.upsert_vectors(batch, batch_size=batch_size)
            pbar.update(len(batch))

    stats = vs.describe_index_stats()
    logger.info(f"Index '{settings.pinecone_index_name}' now contains {stats['total_vector_count']} vectors.")
    logger.info(f"Namespace stats: {stats['namespaces']}")

if __name__ == "__main__":
    main()
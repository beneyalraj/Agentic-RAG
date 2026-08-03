import logging
from pathlib import Path
from config.config import settings
from src.retrieval.bm25 import BM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    chunks_path = settings.chunks_output_path
    output_path = Path("data/bm25_index.pkl")

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return

    logger.info("-- Building BM25 Index --")
    index = BM25Index.build_from_file(chunks_path)

    index.save(output_path)

    logger.info(f"BM25 index saved to {output_path}")
    logger.info(f"Total documents: {len(index.chunks)}")


if __name__ == "__main__":
    main()
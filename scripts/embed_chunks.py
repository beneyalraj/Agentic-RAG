#!/usr/bin/env python3
import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from config.config import settings
from src.retrieval.embeddings import get_embedding_model


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_chunks(filepath: Path):

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    input_path = settings.chunks_output_path
    output_path = Path("data/embedded_chunks.jsonl")

    if not input_path.exists():
        logger.error(f"Chunks file not found: {input_path}")
        return

    model = get_embedding_model(settings.EMBEDDING_MODEL)
    logger.info(f"Using model: {model.model_name} on {model.device}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    batch_size = 32
    batch_texts = []
    batch_metadata = []
    written_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:

        with tqdm(total=total_lines, desc="Embedding chunks") as pbar:
            for chunk in load_chunks(input_path):

                batch_texts.append(chunk["text"])
                chunk_copy = {
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk["metadata"],
                }

                chunk_copy["metadata"]["text"] = chunk["text"]
                batch_metadata.append(chunk_copy)

                if len(batch_texts) >= batch_size:
                    embeddings = model.encode_batch(batch_texts, batch_size=batch_size)

                    for meta, emb in zip(batch_metadata, embeddings):

                        meta["embedding"] = emb.tolist()
                        out_f.write(json.dumps(meta) + "\n")
                        written_count += 1

                    batch_texts = []
                    batch_metadata = []
                    pbar.update(batch_size)

            if batch_texts:
                embeddings = model.encode_batch(batch_texts, batch_size=batch_size)

                for meta, emb in zip(batch_metadata, embeddings):
                    meta["embedding"] = emb.tolist()
                    out_f.write(json.dumps(meta) + "\n")
                    written_count += 1
                pbar.update(len(batch_texts))

    logger.info(f"Embedded {written_count} chunks to {output_path}")


if __name__ == "__main__":
    main()
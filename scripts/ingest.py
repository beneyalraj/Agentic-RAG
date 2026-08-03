import argparse
import hashlib
import logging
import re
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from config.config import settings

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S",)
logger = logging.getLogger(__name__)

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique MD5 hash of the chunk")
    text: str = Field(..., description="The cleaned, chunked text")
    metadata: Dict[str, Any] = Field(..., description="Source, page, chunk index, etc.")

EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL
try:
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, clean_up_tokenization_spaces=True)
    logger.info(f"Loaded tokenizer for {EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.warning(f"Could not load tokenizer for {EMBEDDING_MODEL_NAME}: {e}. Falling back to character count.")
    tokenizer = None

def token_length_function(text: str) -> int:
    if tokenizer is None:
        return len(text)
    return len(tokenizer.tokenize(text))

def text_clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('..', '.')
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line.strip())]
    text = '\n'.join(cleaned_lines)
    return text.strip()

def process_documents(data_dir: Path) -> List[DocumentChunk]:
    supported_extensions = {'.pdf', '.txt', '.md'}
    all_chunks = []

    if not data_dir.exists():
        logger.warning(f"Directory '{data_dir}' not found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=20,
        length_function=token_length_function,
        separators=['\n\n', '\n', '.', ' ', ''],
    )

    for filepath in data_dir.rglob("*"):
        if filepath.suffix.lower() not in supported_extensions:
            continue

        logger.info(f"Processing : {filepath.relative_to(data_dir)}")
        docs = []

        try:
            if filepath.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
            else:
                try:
                    loader = TextLoader(str(filepath), encoding='utf-8')
                    docs = loader.load()
                except UnicodeDecodeError:
                    logger.warning(f"utf-8 failed for {filepath}, trying latin-1...")
                    loader = TextLoader(str(filepath), encoding='latin-1')
                    docs = loader.load()
        except Exception as e:
            logger.error(f"Could not process {filepath.name}: {e}")
            continue

        for doc_idx, doc in enumerate(docs):
            cleaned = text_clean(doc.page_content)
            if not cleaned:
                continue
            splits = text_splitter.split_text(cleaned)

            filing_type = filepath.parent.parent.name
            accession = filepath.parent.name
            source = f"{filing_type}_{accession}"

            for split_idx, split_text in enumerate(splits):
                unique_string = f"{source}_{doc_idx}_{split_idx}"
                chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
                page_num = doc.metadata.get('page', 1) if filepath.suffix.lower() == '.pdf' else 1

                metadata = {
                    "source": source,
                    "page": page_num,
                    "chunk_index": split_idx,
                    "total_chunks_in_doc": len(splits)
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=split_text,
                    metadata=metadata
                )
                all_chunks.append(chunk)

    return all_chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.cleaned_data_dir,
        help='Path to the data directory (default from config)'
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.chunks_output_path,
        help='Save chunks as JSON Lines (.jsonl)'
    )
    args = parser.parse_args()

    logger.info("-- Starting Ingestion Process --")
    chunks = process_documents(args.data_dir)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
        logger.info(f"Saved {len(chunks)} chunks to {args.output}")

    logger.info(f"--- Ingestion Complete --- Total chunks: {len(chunks)}")
    if chunks:
        print("\nSample Output:")
        print(chunks[0].model_dump_json(indent=2))

if __name__ == "__main__":
    main()
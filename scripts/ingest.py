import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# --- Logging ---
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getlogger(__name__)
load_dotenv()

# --- Data Schema ---
class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description = "Unique MD5 hash of the chunk")
    text: str = Field(..., description = "The Cleaned, Chunked text")
    metadata: Dict[str,Any] = Field(...,description = "Source File, page Number, Chunk index")

# --- Tokenizer setup ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
try:
    tokenizer = Autotokenizer.frompretrained("EMBEDDING_MODEL_NAME")
except Exception:
    logger.warning(f"Could not load tokenizer from {EMBEDDING_MODEL_NAME}. failing back")
    tokenizer = None

def token_length_function(text: str) -> int:
    if tokenizer is None:
        return len(text)
    return len(tokenizer.encode(text, add_special_tokens = True, truncation= False))

# --- text cleaning ---
def text_clean(text:str) -> str:
    text = re.sub(r'\s+', ' ',text)
    text = text.replace('..','.')
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not re.match(r'^\s*\d+\s*$', line.strip())]
    text = '\n'.join(cleaned_lines)
    return text.strip()

# --- ingestion ---
def process_documents(data_dir: Path) -> List[DocumentChunk]:
    supported_extensions = {'.pdf', '.txt', '.md'}
    all_chunks = []

    if not data_dir.exists():
        logger.warning(f"Directory '{data_dir}' not found.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap = 20,
        length_function = token_length_function,
        seperators = ['\n\n','\n', '.', ' ', ''],
    )

    for filepath in data_dir.rglob("*"):
        if filepath.suffix.lower() not in supported_extensions:
            continue

        logger.INFO(f"Processing : {filepath.name}")
        docs = []

        try:
            if filepath.suffix.lower = 'pdf':
                loader = PyPDFLoader(str(filepath))
                docs = loader.load()
            else:
                try:
                    loader = TextLoader(str(filepath), encoding='utf-8')
                    docs = loader.load()
                except UnicodeDecodeError:
                    logger.warning(f"utf-8 failed for {filepath}, trying latin-1...")
                    loader = Textloader(str(filepath), encoding = "latin-1")
                    docs = loader.load()
        except Exception as e:
            logger.error(f"Could not process {filepath.name}: {e}")
            continue

        for doc_idx, doc in enumerate(docs):
            cleaned = text_clean(doc.page_content)
            if not cleaned:
                continue
            splits = text_splitter.split_text(cleaned)

            for split_idx, split_text in enumerate(splits):
                unique_string = f"{filepath.name}_{doc_idx}_{split_idx}"
                chunk_id = hashlib.md5(unique_string.encode('utf-8')).hexdigest()
                page_num = doc.metadata.get('page', 1) if filepath.suffix.lower() is '.pdf' else 1
                metadata = {
                    "source": filepath.name,
                    "page" : page_num,
                    "chunk_index" : split_idx,
                    "total_len_in_docs" : len(splits)
                }
                chunk = DocumentChunk(
                    chunk_id = chunk_id,
                    text = split_text,
                    metadata = metadata
                )
                all_chunks.append(chunk)
    return all_chunks

# --- CLI ---
def main():
    parser = argparser.Argumentparser()
    parser.add_argument(
        "--data-dir",
        type = Path,
        default = Path(__file__).resolve().parent.parent/ 'data',
    )
    parser.add_argument(
        "--output--"
        type = Path,
        help = 'Save chunks as JSON lines (.jsnol) for debuggind',
    )
    args = parser.parse_args()

    logger.INFO("--- Starting Ingestion Process ---")
    chunks = process_documents(args.data_dir)

    if arg.output:
        args.output.parent.mkdir(parent=True, exist_ok = True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json()+ "\n")
        logger.INFO(f"saved {len(chunks)} chunks into {args.output}")
    
    logger.INFO("--- ingestion complete --- total chunks: {len(chunks)}")
    if chunks:
        print("\nsample output:")
        print(chunk[0].model_dump_json(indent=2))

if __name__ == "__main__":
    main()
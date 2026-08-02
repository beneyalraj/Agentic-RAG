import re
import logging
from pathlib import Path
from bs4 import BeautifulSoup
from config.config import settings

RAW_DIR = settings.raw_data_dir
CLEAN_DIR = settings.cleaned_data_dir
SUPPORTED_EXTENSIONS = {".html", ".htm", ".txt"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_primary_document_text(filepath: Path, filing_type: str) -> str:

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    doc_pattern = re.compile(r'<DOCUMENT>(.*?)</DOCUMENT>', re.DOTALL | re.IGNORECASE)
    documents = doc_pattern.findall(content)

    if not documents:
        logger.debug(f"No <DOCUMENT> tags found in {filepath.name}, trying fallback.")
        fallback = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL | re.IGNORECASE)

        if fallback:
            return fallback.group(1).strip()
        return ""

    candidates = []

    for doc in documents:
        type_match = re.search(r'<TYPE>\s*([^\s<]+)', doc, re.IGNORECASE)
        seq_match = re.search(r'<SEQUENCE>\s*(\d+)', doc, re.IGNORECASE)
        doc_type = type_match.group(1) if type_match else None
        seq_num = int(seq_match.group(1)) if seq_match else None

        text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL | re.IGNORECASE)
        if not text_match:
            continue
        text_content = text_match.group(1).strip()

        if len(text_content) < 200:
            continue

        if doc_type and doc_type.upper() == filing_type:
            candidates.append((seq_num or 999, text_content, doc_type))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        logger.debug(f"Picked {candidates[0][2]} (seq {candidates[0][0]}) from {len(candidates)} candidates")

        return candidates[0][1]

    logger.debug(f"No matching document found for {filing_type}; using fallback.")
    first_text = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL | re.IGNORECASE)

    if first_text:

        fallback_text = first_text.group(1).strip()
        if len(fallback_text) > 500:
            return fallback_text

    return ""


def clean_text(raw_text: str) -> str:

    soup = BeautifulSoup(raw_text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    hidden_style_patterns = ["display:none", "visibility:hidden"]

    for tag in soup.find_all():
        if tag.attrs is None:
            continue

        if tag.name and tag.name.startswith("ix:"):
            tag.decompose()
            continue

        style = tag.get("style", "")
        if style and isinstance(style, str):
            if any(pattern in style.replace(" ", "") for pattern in hidden_style_patterns):
                tag.decompose()
                continue

    for tag in soup.find_all(class_=["hidden", "ix-hidden"]):
        if tag.attrs is None:
            continue
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_filings():
    if not RAW_DIR.exists():
        logger.error(f"RAW_DIR '{RAW_DIR}' does not exist.")
        return

    total_files = 0
    skipped_files = 0
    failed_files = 0

    for filepath in RAW_DIR.rglob("*"):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped_files += 1
            continue

        filing_type = filepath.parent.parent.name
        if filing_type not in {"10-K", "10-Q", "8-K"}:
            logger.warning(f"Unknown filing type '{filing_type}' for {filepath}")
            filing_type = None

        relative_path = filepath.relative_to(RAW_DIR)
        clean_relative_path = relative_path.with_suffix(".txt")
        clean_filepath = CLEAN_DIR / clean_relative_path

        if clean_filepath.exists():
            logger.debug(f"Skipping already cleaned: {clean_filepath}")
            continue

        try:
            extracted_text = extract_primary_document_text(filepath, filing_type)
            if not extracted_text:
                logger.warning(f"No text extracted from {filepath} (type={filing_type})")
                failed_files += 1
                continue

            clean_text_content = clean_text(extracted_text)
            if len(clean_text_content) < 500:
                logger.warning(f"Extracted text too short ({len(clean_text_content)} chars) for {filepath} – skipping")
                failed_files += 1
                continue

            clean_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(clean_filepath, "w", encoding="utf-8") as f:
                f.write(clean_text_content)

            total_files += 1
            if total_files % 5 == 0:
                logger.info(f"Progress: Cleaned {total_files} filings...")

        except Exception as e:
            logger.error(f"Failed to clean {filepath}: {e}")
            failed_files += 1

    logger.info(f"Cleanup completed. Files cleaned: {total_files}, Files skipped (unsupported ext): {skipped_files}, Files failed: {failed_files}, Clean output folder: {CLEAN_DIR}")


if __name__ == "__main__":
    clean_filings()
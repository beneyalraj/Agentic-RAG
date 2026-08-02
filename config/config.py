from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    raw_data_dir: Path = Path("data/financial/sec-edgar-filings")
    cleaned_data_dir: Path = Path("data/financial/cleaned")
    chunks_output_path: Path = Path("data/chunks.jsonl")

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" 

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
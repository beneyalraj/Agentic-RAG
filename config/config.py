from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    raw_data_dir: Path = Path("data/financial/sec-edgar-filings")
    cleaned_data_dir: Path = Path("data/financial/cleaned")
    chunks_output_path: Path = Path("data/chunks.jsonl")
    sql_output_path: Path = Path("data/financial/company_facts_raw.json")
    db_path: Path = Path("data/financials.db")
    BASE_URL: str = "http://localhost:8000/api/v1"

    target_cik: str = "0000019617"
    sql_url: str= "https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "financial-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    groq_api_key: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    max_iterations: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
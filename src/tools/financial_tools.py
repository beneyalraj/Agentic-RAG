import ast
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict
import asyncio 

import sqlparse

from config.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import PineconeVectorStore
from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import get_embedding_model

logger = logging.getLogger(__name__)

_retriever = None
_chunks_metadata = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        logger.info("Initialising hybrid retriever...")
        embedder = get_embedding_model()
        dense_store = PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
            dimension=384,
        )
        dense_store.create_index_if_not_exists()
        bm25_path = Path("data/bm25_index.pkl")
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run build_bm25.py first.")
        sparse_store = BM25Index.load(bm25_path)
        _retriever = HybridRetriever(
            dense_retriever=dense_store,
            sparse_retriever=sparse_store,
            embed_query=lambda q: embedder.encode_single(q).tolist(),
        )
    return _retriever


def _get_db_connection() -> sqlite3.Connection:
    db_path = Path("data/financials.db").absolute()
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def get_chunks_metadata() -> Dict[str, Any]:
    global _chunks_metadata
    if _chunks_metadata is None:
        chunks_file = settings.chunks_output_path
        if not chunks_file.exists():
            return {"error": "Chunks file not found"}
        sources = set()
        filing_types = set()
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    chunk = json.loads(line)
                    src = chunk.get("metadata", {}).get("source", "")
                    if src:
                        sources.add(src)
                        ft = src.split("_")[0] if "_" in src else "unknown"
                        filing_types.add(ft)
            _chunks_metadata = {
                "total_chunks": sum(1 for _ in open(chunks_file)),
                "unique_sources": len(sources),
                "filing_types": list(filing_types),
                "sample_sources": list(sources)[:5],
            }
        except Exception as e:
            _chunks_metadata = {"error": str(e)}
    return _chunks_metadata

def tool_calculator(expression: str) -> str:
    try:
        result = safe_eval(expression)
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"
        
def is_read_only_sql(sql: str) -> bool:
    parsed = sqlparse.parse(sql)
    if not parsed:
        return False
    return parsed[0].get_type().upper() == "SELECT"


def safe_eval(expr: str) -> float:
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd,
    }
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise ValueError(f"Invalid expression: {expr}")
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")
    code = compile(tree, '<string>', 'eval')
    safe_globals = {'__builtins__': None, 'pow': pow, 'abs': abs}
    return float(eval(code, safe_globals))


async def tool_search_filings(query: str, top_k: int = 5) -> str:
    try:
        retriever = get_retriever()
        results = await asyncio.to_thread(retriever.hybrid_search, query=query, top_k=top_k)
        if not results:
            return "No results found."
        output = []
        for idx, res in enumerate(results, 1):
            output.append(
                f"Result {idx} (RRF score: {res['rrf_score']:.4f})\n"
                f"Source: {res['metadata'].get('source', 'Unknown')}\n"
                f"Snippet: {res['text'][:300]}...\n"
            )
        return "\n".join(output)
    except Exception as e:
        return f"Error in search_filings: {e}"



def tool_sql_query(sql: str) -> str:
    if not is_read_only_sql(sql):
        return "Error: Only SELECT queries are allowed for security."
    conn = _get_db_connection()
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            valid_metrics = conn.execute(
                "SELECT DISTINCT metric_name FROM financial_metrics ORDER BY metric_name"
            ).fetchall()
            metric_list = ", ".join(r["metric_name"] for r in valid_metrics)
            return (
                f"Query returned no results. This is likely because the metric_name doesn't exist. "
                f"Valid metric_name values are: {metric_list}"
            )
        col_names = [desc[0] for desc in cursor.description]
        result = " | ".join(col_names) + "\n"
        result += "-" * len(result) + "\n"
        for row in rows:
            result += " | ".join(str(val) for val in row) + "\n"
        return result.strip()
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        conn.close()

def tool_financial_ratio_calculator(ratio_name: str, **kwargs) -> str:
    try:
        ratio = ratio_name.lower()
        if ratio == "roe":
            ni, eq = kwargs.get("net_income"), kwargs.get("equity")
            if ni is None or eq is None or eq == 0:
                return "Error: Need net_income and equity (positive equity)."
            return f"ROE = {ni / eq:.4f}"
        elif ratio == "debt_equity":
            liabilities, eq = kwargs.get("total_liabilities"), kwargs.get("equity")
            if liabilities is None or eq is None or eq == 0:
                return "Error: Need total_liabilities and equity."
            return f"Debt-to-Equity = {liabilities / eq:.4f}"
        elif ratio == "roa":
            ni, assets = kwargs.get("net_income"), kwargs.get("assets")
            if ni is None or assets is None or assets == 0:
                return "Error: Need net_income and assets."
            return f"ROA = {ni / assets:.4f}"
        elif ratio == "current_ratio":
            ca, cl = kwargs.get("current_assets"), kwargs.get("current_liabilities")
            if ca is None or cl is None or cl == 0:
                return "Error: Need current_assets and current_liabilities."
            return f"Current Ratio = {ca / cl:.4f}"
        elif ratio == "quick_ratio":
            ca, inv, cl = kwargs.get("current_assets"), kwargs.get("inventory", 0), kwargs.get("current_liabilities")
            if ca is None or cl is None or cl == 0:
                return "Error: Need current_assets and current_liabilities."
            return f"Quick Ratio = {(ca - inv) / cl:.4f}"
        else:
            supported = "ROE, Debt_Equity, ROA, Current_Ratio, Quick_Ratio"
            return f"Error: Unsupported ratio '{ratio_name}'. Supported: {supported}."
    except Exception as e:
        return f"Error: {e}"


def tool_get_filing_metadata() -> str:
    meta = get_chunks_metadata()
    conn = _get_db_connection()
    try:
        total_rows = conn.execute("SELECT COUNT(*) as total FROM financial_metrics").fetchone()["total"]
        row = conn.execute("SELECT MIN(period_end) as min_date, MAX(period_end) as max_date FROM financial_metrics").fetchone()
        min_date, max_date = row["min_date"], row["max_date"]
        metrics = [r["metric_name"] for r in conn.execute("SELECT DISTINCT metric_name FROM financial_metrics ORDER BY metric_name").fetchall()]
    except Exception as e:
        return f"Error querying DB: {e}"
    finally:
        conn.close()

    summary = f"""
        Data Overview:
        - Text Chunks: {meta.get('total_chunks', 0)} chunks from {meta.get('unique_sources', 0)} unique filing sources.
        - Filing Types: {', '.join(meta.get('filing_types', ['N/A']))}
        - Sample sources: {', '.join(meta.get('sample_sources', ['N/A']))}

        Financial Metrics (SQLite):
        - Total records: {total_rows}
        - Date range: {min_date} to {max_date}
        - Available metrics: {', '.join(metrics)}
        """
    return summary.strip()
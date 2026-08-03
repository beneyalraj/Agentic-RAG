import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from config.config import settings

RAW_FACTS_PATH = settings.sql_output_path
DB_PATH = settings.db_path
TABLE_NAME = "financial_metrics"

CONCEPTS = {
    "NetIncomeLoss": "NetIncomeLoss",
    "Revenues": "Revenues",
    "EarningsPerShareDiluted": "EarningsPerShareDiluted",
    "EarningsPerShareBasic": "EarningsPerShareBasic",
    "Assets": "Assets",
    "Liabilities": "Liabilities",
    "StockholdersEquity": "StockholdersEquity",
    "OperatingIncomeLoss": "OperatingIncomeLoss",
}

FALLBACK_CONCEPTS = {
    "NetIncomeLoss": [
        "NetIncomeLossAvailableToCommonStockholders",
        "NetIncomeLossAttributableToParent"
    ],

    "Revenues": ["RevenueFromContractWithCustomerIncludingAssessedTax"],

    "OperatingIncomeLoss": [
        "OperatingIncomeLossExcludingOperatingExpenses",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ]
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data() -> Dict[str, Any]:
    if not RAW_FACTS_PATH.exists():

        raise FileNotFoundError(
            f"Raw facts file not found at {RAW_FACTS_PATH}. "
            "Run fetch_company_facts.py first."
        )
    with open(RAW_FACTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_concept_values(data: Dict[str, Any], concept: str, fallbacks: Optional[List[str]] = None) -> List[Dict[str, Any]]:

    facts = data.get("facts", {})
    us_gaap = facts.get("us-gaap", {})

    if concept in us_gaap:
        units = us_gaap[concept].get("units", {})
        unit = "USD" if "USD" in units else next(iter(units))
        return units.get(unit, [])

    if fallbacks:
        for fb in fallbacks:
            if fb in us_gaap:
                units = us_gaap[fb].get("units", {})
                unit = "USD" if "USD" in units else next(iter(units))
                logger.info(f"Using fallback concept '{fb}' for '{concept}'")
                return units.get(unit, [])

    logger.warning(f"Concept '{concept}' not found in data (no fallbacks matched)")
    return []


def flatten_to_rows(concept_name: str, values: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    rows = []
    for entry in values:
        form = entry.get("form", "")
        if form not in ("10-K", "10-Q"):
            continue

        rows.append({
            "metric_name": concept_name,
            "value": entry.get("val"),
            "unit": entry.get("unit", "USD"),
            "period_end": entry.get("end"),
            "form_type": form,
            "filed_date": entry.get("filed"),
        })
    return rows


def create_table(conn: sqlite3.Connection) -> None:

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            period_end TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filed_date TEXT NOT NULL,
            UNIQUE(metric_name, period_end, form_type)
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_metric_period ON {TABLE_NAME} (metric_name, period_end)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_form_type ON {TABLE_NAME} (form_type)")
    conn.commit()

def insert_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> int:
    cursor = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            cursor.execute(f"""
                INSERT OR IGNORE INTO {TABLE_NAME}
                (metric_name, value, unit, period_end, form_type, filed_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["metric_name"],
                row["value"],
                row["unit"],
                row["period_end"],
                row["form_type"],
                row["filed_date"],
            ))
            inserted += cursor.rowcount
        except Exception as e:
            logger.error(f"Failed to insert row {row}: {e}")
    conn.commit()
    return inserted


def print_summary(conn: sqlite3.Connection) -> None:

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT metric_name, COUNT(*), MIN(period_end), MAX(period_end)
        FROM {TABLE_NAME}
        GROUP BY metric_name
        ORDER BY metric_name
    """)
    results = cursor.fetchall()

    total_rows = sum(r[1] for r in results)
    logger.info("SUMMARY")
    logger.info(f"Total rows in database: {total_rows}")

    for metric, count, min_date, max_date in results:
        logger.info(f"{metric:20} | {count:5} rows | {min_date} → {max_date}")


def main():
    logger.info("Loading raw Company Facts data...")
    data = load_raw_data()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))

    create_table(conn)

    total_inserted = 0
    for display_name, concept in CONCEPTS.items():
        logger.info(f"Extracting '{concept}'...")
        fallbacks = FALLBACK_CONCEPTS.get(display_name)
        values = extract_concept_values(data, concept, fallbacks)
        if not values:
            logger.warning(f"No data found for '{concept}'. Skipping.")
            continue

        rows = flatten_to_rows(display_name, values)
        if not rows:
            logger.info(f"No 10-K/10-Q rows for '{concept}'. Skipping.")
            continue

        inserted = insert_rows(conn, rows)
        total_inserted += inserted
        logger.info(f"Inserted {inserted} rows for '{display_name}'")

    print_summary(conn)

    conn.close()
    logger.info(f"Database saved to {DB_PATH}")

if __name__ == "__main__":
    main()
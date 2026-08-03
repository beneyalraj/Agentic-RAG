import json
import logging
from pathlib import Path
import requests
from config.config import settings

URL = str(settings.sql_url).format(CIK=settings.target_cik)
OUTPUT_PATH = settings.sql_output_path

HEADERS = {"User-Agent": "AgenticRAGAssistant contact@yourapp.com", "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():

    settings.sql_output_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.sql_output_path.exists():
        logger.info(f"Using cached data from {settings.sql_output_path}")
        return

    logger.info(f"Fetching Company Facts for CIK {settings.target_cik} from SEC API")
    try:
        response = requests.get(URL, headers=HEADERS, timeout=30)
        response.raise_for_status()

        data = response.json()
        with open(settings.sql_output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved raw data to {settings.sql_output_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data: {e}")
        raise

if __name__ == "__main__":
    main()
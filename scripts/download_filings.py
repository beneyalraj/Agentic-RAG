import os
import sys
from dotenv import load_dotenv
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader

CIK = "0000019617"
OUTPUT_DIR = Path("data/financial")

load_dotenv()
COMPANY_NAME = os.getenv("COMPANY_NAME")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")

logging.basicConfig(level=logging.INFO, format= "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        dl = Downloader(COMPANY_NAME, EMAIL_ADDRESS, str(OUTPUT_DIR))
        logger.info(f"Downloading files for {CIK} into {OUTPUT_DIR}")

        logger.info("Downloading latest 10k files")
        dl.get("10-K", CIK, limit=1)

        logger.info("Downloading last 10 10-Quater filings")
        dl.get("10-Q", CIK, limit=10)

        logger.info("Downloading last 20 8k filings")
        dl.get("8-K", CIK, limit=20)

        logger.info("Downloaded files successfully")

    except Exception as e:
        logger.error(f"Download failed {e}")

if __name__ == "__main__":
    main()
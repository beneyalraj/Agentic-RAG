import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.config import settings
import requests
import json

BASE_URL = settings.BASE_URL

def test_query():
    payload = {
        "query": "What was JPMorgan's ROE using their most recent net income and stockholders equity?"
    }
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Answer:", data["answer"])
        print("Tools called:", data["tool_calls_made"])
        print("Thread ID:", data["thread_id"])
    else:
        print("Error:", response.text)


if __name__ == "__main__":
    test_query()
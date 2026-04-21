import json
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "data"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    docs = list(read_jsonl(base / "raw_documents_10pct.jsonl"))
    tickets = list(read_jsonl(base / "raw_support_tickets_10pct.jsonl"))
    print("docs:", len(docs))
    print("tickets:", len(tickets))
    print("sample doc keys:", docs[0].keys())
    print("sample ticket keys:", tickets[0].keys())

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient


def _pick_database(client: MongoClient, requested_db: str | None) -> str:
    if requested_db:
        return requested_db

    env_db = os.getenv("MONGO_DB_NAME")
    if env_db:
        return env_db

    names = [name for name in client.list_database_names() if name not in {"admin", "local", "config"}]
    if not names:
        raise RuntimeError("No user database found. Provide --db or set MONGO_DB_NAME.")
    return names[0]


def run_smoke_test(uri: str, db_name: str | None) -> dict[str, Any]:
    with MongoClient(uri, serverSelectionTimeoutMS=7000) as client:
        ping = client.admin.command("ping")
        selected_db_name = _pick_database(client, db_name)
        db = client[selected_db_name]

        collections = db.list_collection_names()
        collection_counts = {}
        for name in collections[:10]:
            collection_counts[name] = db[name].estimated_document_count()

        return {
            "status": "ok",
            "ping": ping,
            "mongo_uri": uri,
            "database": selected_db_name,
            "collections": collections,
            "collection_count_preview": collection_counts,
        }


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="MongoDB smoke test for hackathon backend")
    parser.add_argument("--uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    try:
        result = run_smoke_test(args.uri, args.db)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "status": "error",
                    "mongo_uri": args.uri,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.insight import build_insight  # noqa: E402
from app.services.mongo_executor import MongoExecutor  # noqa: E402
from app.services.planner import build_plan  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run planner -> query -> insight dry run")
    parser.add_argument(
        "--question",
        required=True,
        help="User question to process",
    )
    args = parser.parse_args()

    plan, semantic_candidates = build_plan(args.question, {"strict": True})
    executor = MongoExecutor()
    query_result = executor.run_plan_with_repair(plan, semantic_candidates=semantic_candidates)

    query_errors = [attempt.error for attempt in query_result.attempts if attempt.error]
    stats_payload = {
        "anomalies": [],
        "note": "placeholder stats until stats service is integrated",
        "query_status": query_result.status,
        "query_row_count": query_result.row_count,
        "query_corrected_automatically": query_result.corrected_automatically,
        "query_errors": query_errors,
    }
    insight = build_insight(plan, query_result.rows, stats_payload)

    result = {
        "question": args.question,
        "planner": {
            "plan": plan.model_dump(),
            "semantic_candidates": semantic_candidates,
        },
        "query": query_result.model_dump(),
        "insight": insight.model_dump(),
    }

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

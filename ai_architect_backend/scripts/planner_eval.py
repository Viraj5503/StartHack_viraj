import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import IntentCategory  # noqa: E402
from app.services.planner import build_plan  # noqa: E402


def load_cases(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_cases(cases: list[dict[str, str]], strict: bool) -> tuple[int, int, list[str]]:
    passes = 0
    details: list[str] = []

    for idx, case in enumerate(cases, start=1):
        question = case["question"]
        expected = IntentCategory(case["expected_intent"])
        plan, semantic_candidates = build_plan(question, {"strict": strict})

        matched_intent = plan.intent == expected
        has_collection = len(plan.required_collections) > 0

        chart_expected = expected in {
            IntentCategory.comparison,
            IntentCategory.trend_drift,
            IntentCategory.hypothesis,
            IntentCategory.anomaly_check,
        }
        chart_ok = (not chart_expected) or plan.chart_needed

        case_pass = matched_intent and has_collection and chart_ok
        if case_pass:
            passes += 1

        details.append(
            (
                f"[{idx:02d}] {'PASS' if case_pass else 'FAIL'} | "
                f"expected={expected.value} predicted={plan.intent.value} "
                f"| collections={[c.value for c in plan.required_collections]} "
                f"| chart_needed={plan.chart_needed} "
                f"| semantic_candidates={len(semantic_candidates)}"
            )
        )

    return passes, len(cases), details


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate planner intent and structure on predefined prompt set")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "resources" / "planner_eval_cases.json"),
        help="Path to planner evaluation cases JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run planner with strict context flag",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.8,
        help="Minimum pass ratio required for zero exit code",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Cases file not found: {cases_path}")
        return 2

    cases = load_cases(cases_path)
    passes, total, details = evaluate_cases(cases, args.strict)

    print("Planner Evaluation Results")
    print("==========================")
    for line in details:
        print(line)

    ratio = (passes / total) if total else 0.0
    print("--------------------------")
    print(f"Passed: {passes}/{total} ({ratio:.1%})")
    print(f"Threshold: {args.fail_below:.1%}")

    return 0 if ratio >= args.fail_below else 1


if __name__ == "__main__":
    raise SystemExit(main())

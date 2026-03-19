import re
from typing import Any

from app.config import get_settings
from app.schemas import InsightResponse, IntentCategory
from app.services.llm_gateway import ClaudeGateway


def _chart_for_intent(intent: IntentCategory) -> dict[str, Any]:
    if intent == IntentCategory.trend_drift:
        return {
            "type": "line",
            "x": "uploadDate",
            "y": "value",
            "title": "Trend over time",
        }
    if intent == IntentCategory.comparison:
        return {
            "type": "bar",
            "x": "group",
            "y": "metric",
            "title": "Comparison view",
        }
    if intent == IntentCategory.anomaly_check:
        return {
            "type": "scatter",
            "x": "index",
            "y": "value",
            "title": "Anomaly scan",
        }
    return {
        "type": "table",
        "title": "Result overview",
    }


def _follow_up_for_intent(intent: IntentCategory) -> list[str]:
    if intent == IntentCategory.comparison:
        return [
            "Show the same comparison only for Tuesday.",
            "Compare the same materials on another machine.",
            "Run significance test with stricter confidence level.",
        ]
    if intent == IntentCategory.trend_drift:
        return [
            "Limit the trend to the last 30 days.",
            "Highlight only values near boundary violations.",
            "Compare this trend against another site.",
        ]
    if intent == IntentCategory.validation_compliance:
        return [
            "List only records that violate the limit.",
            "Show compliance by machine.",
            "Generate a compliance summary report.",
        ]
    return [
        "Apply a narrower filter window.",
        "Compare against another machine or site.",
        "Generate a reusable report from this result.",
    ]


def _normalize_str_list(value: Any, fallback: list[str], max_items: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return fallback

    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        return fallback

    if max_items is not None:
        return normalized[:max_items]
    return normalized


def _normalize_chart_config(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return fallback

    allowed_types = {"line", "bar", "scatter", "table"}
    chart_type = str(value.get("type", fallback.get("type", "table"))).strip().lower()
    if chart_type not in allowed_types:
        chart_type = str(fallback.get("type", "table"))

    normalized: dict[str, Any] = {
        "type": chart_type,
        "title": str(value.get("title") or fallback.get("title") or "Result overview").strip(),
    }

    def normalize_chart_field_name(field_name: str) -> str:
        token = field_name.strip().lstrip("$")
        token = re.sub(r"^(?:_tests|tests|Tests|valuecolumns_migrated|values|Values)\.", "", token)
        normalized_token = re.sub(r"[^a-z0-9]+", "", token.lower())

        if normalized_token in {"testdate", "createdat", "timestamp", "time", "uploaddate"}:
            return "uploadDate"
        if normalized_token in {"tensilestrength", "strength", "value", "values"}:
            return "value"
        return token

    x_value = value.get("x")
    y_value = value.get("y")
    if isinstance(x_value, str) and x_value.strip():
        normalized["x"] = normalize_chart_field_name(x_value)
    elif isinstance(fallback.get("x"), str):
        normalized["x"] = normalize_chart_field_name(str(fallback["x"]))

    if isinstance(y_value, str) and y_value.strip():
        normalized["y"] = normalize_chart_field_name(y_value)
    elif isinstance(fallback.get("y"), str):
        normalized["y"] = normalize_chart_field_name(str(fallback["y"]))

    if chart_type == "table":
        normalized.pop("x", None)
        normalized.pop("y", None)

    return normalized


def _sample_rows(rows: list[dict[str, Any]], max_rows: int = 8, max_keys: int = 10) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, dict):
            continue
        keys = list(row.keys())[:max_keys]
        sample.append({key: row.get(key) for key in keys})
    return sample


def _build_insight_mock(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse:
    row_count = len(rows)
    anomalies = stats.get("anomalies", []) if isinstance(stats, dict) else []

    summary = [
        f"Processed {row_count} rows for intent '{plan.intent.value}'.",
        f"Applied operations: {', '.join(op.value for op in plan.operations) or 'none' }.",
        "Results are ready for chart rendering and follow-up investigation.",
    ]

    anomaly_notes = []
    if isinstance(anomalies, list) and anomalies:
        anomaly_notes.append(f"Detected {len(anomalies)} potential anomalies from provided stats.")
    elif plan.intent == IntentCategory.anomaly_check:
        anomaly_notes.append("No anomaly list was provided by the stats service yet.")
    else:
        anomaly_notes.append("No major anomaly signal detected in this response.")

    recommendation = (
        "Use the follow-up suggestions to narrow filters, then save this as a reusable template if the result is useful."
    )

    audit_log = [
        "Insight generated from query output plus statistics payload.",
        f"Input row count: {row_count}",
        f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
    ]

    return InsightResponse(
        summary_3_sentences=summary,
        anomaly_notes=anomaly_notes,
        recommendation=recommendation,
        follow_up_questions=_follow_up_for_intent(plan.intent),
        chart_config=_chart_for_intent(plan.intent),
        audit_log=audit_log,
    )


def _build_insight_llm(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse | None:
    settings = get_settings()
    if settings.llm_provider.lower() != "anthropic":
        return None

    gateway = ClaudeGateway()
    if not gateway.is_ready():
        return None

    row_count = len(rows)
    fallback = _build_insight_mock(plan, rows, stats)
    sample_rows = _sample_rows(rows)
    stats_keys = sorted(stats.keys()) if isinstance(stats, dict) else []

    system_prompt = (
        "You are an AI data analyst for materials testing. "
        "Return strict JSON only and keep statements grounded in provided data."
    )
    user_prompt = (
        "Generate insight JSON for this query output.\n"
        "Required schema:\n"
        "{\n"
        '  "summary_3_sentences": string[3],\n'
        '  "anomaly_notes": string[],\n'
        '  "recommendation": string,\n'
        '  "follow_up_questions": string[],\n'
        '  "chart_config": {"type": "line|bar|scatter|table", "x": string?, "y": string?, "title": string},\n'
        '  "audit_log": string[]\n'
        "}\n"
        "Rules:\n"
        "- summary_3_sentences must contain exactly 3 concise sentences.\n"
        "- follow_up_questions should contain 3 to 5 actionable items.\n"
        "- If row_count is 0, explicitly state data is insufficient and avoid fabricated findings.\n"
        "- Never include markdown.\n"
        f"Intent: {plan.intent.value}\n"
        f"Operations: {[op.value for op in getattr(plan, 'operations', [])]}\n"
        f"Question: {getattr(plan, 'user_question', '')}\n"
        f"Row count: {row_count}\n"
        f"Sample rows: {sample_rows}\n"
        f"Stats keys: {stats_keys}\n"
        f"Stats payload: {stats if isinstance(stats, dict) else {}}\n"
    )

    result = gateway.generate_json(
        model=settings.anthropic_model_insight,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=1400,
        temperature=0.1,
    )
    if not result:
        return None

    summary = _normalize_str_list(result.get("summary_3_sentences"), fallback.summary_3_sentences, max_items=3)
    if len(summary) < 3:
        summary = (summary + fallback.summary_3_sentences)[:3]

    anomaly_notes = _normalize_str_list(result.get("anomaly_notes"), fallback.anomaly_notes, max_items=5)
    follow_ups = _normalize_str_list(result.get("follow_up_questions"), fallback.follow_up_questions, max_items=5)
    if len(follow_ups) < 3:
        follow_ups = (follow_ups + fallback.follow_up_questions)[:3]

    recommendation_raw = result.get("recommendation")
    recommendation = (
        recommendation_raw.strip()
        if isinstance(recommendation_raw, str) and recommendation_raw.strip()
        else fallback.recommendation
    )

    chart_config = _normalize_chart_config(result.get("chart_config"), fallback.chart_config)
    audit_log = _normalize_str_list(result.get("audit_log"), fallback.audit_log, max_items=8)

    try:
        return InsightResponse(
            summary_3_sentences=summary,
            anomaly_notes=anomaly_notes,
            recommendation=recommendation,
            follow_up_questions=follow_ups,
            chart_config=chart_config,
            audit_log=audit_log,
        )
    except Exception:  # noqa: BLE001
        return None


def build_insight(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse:
    settings = get_settings()
    if settings.insight_mode.lower() == "llm":
        llm_insight = _build_insight_llm(plan, rows, stats)
        if llm_insight is not None:
            return llm_insight

        fallback = _build_insight_mock(plan, rows, stats)
        fallback.audit_log.append("Insight mode fallback: LLM output unavailable or invalid.")
        return fallback

    return _build_insight_mock(plan, rows, stats)

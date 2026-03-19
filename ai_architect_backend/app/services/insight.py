import math
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


_NUMBER_PATTERN = re.compile(r"^[-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?$")
_LIMIT_TOKENS = {"limit", "limits", "threshold", "permissible", "shutdown", "bound", "bounds"}
_UPPER_LIMIT_TOKENS = {"upper", "maximum", "max", "limit", "ceiling", "permissible"}
_LOWER_LIMIT_TOKENS = {"lower", "minimum", "min", "floor", "threshold", "shutdown"}
_FAMILY_TOKENS = {"force", "strength", "yield", "stress", "thickness", "diameter", "extension", "strain"}
_STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "our",
    "internal",
    "within",
    "against",
    "as",
    "per",
    "check",
    "validate",
    "values",
    "value",
}

_TERM_TO_KEY_HINTS: dict[str, list[str]] = {
    "tensile strength": [
        "Maximum force",
        "Max. permissible force at end of test",
        "Upper yield point",
        "Upper force limit",
        "Force shutdown threshold",
    ],
    "maximum force": [
        "Maximum force",
        "Max. permissible force at end of test",
        "Upper force limit",
        "Force shutdown threshold",
    ],
    "force": [
        "Maximum force",
        "Max. permissible force at end of test",
        "Upper force limit",
        "Force shutdown threshold",
    ],
    "wall thickness": ["Wall thickness", "SPECIMEN_THICKNESS", "Specimen thickness after break"],
    "specimen thickness": ["SPECIMEN_THICKNESS", "Specimen thickness after break", "Wall thickness"],
    "thickness": ["Wall thickness", "SPECIMEN_THICKNESS", "Specimen thickness after break"],
    "yield": ["Upper yield point", "Lower yield point", "Upper force limit"],
}


def _normalize_compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokenize_words(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.count(",") > 0 and text.count(".") > 0:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") > 0 and text.count(".") == 0:
        text = text.replace(",", ".")

    if not _NUMBER_PATTERN.match(text):
        return None

    try:
        parsed = float(text)
    except Exception:  # noqa: BLE001
        return None

    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _is_limit_key(key: str) -> bool:
    words = set(_tokenize_words(key))
    return bool(words & _LIMIT_TOKENS)


def _extract_numeric_key_values(rows: list[dict[str, Any]]) -> tuple[list[dict[str, float]], dict[str, list[float]]]:
    row_numeric_maps: list[dict[str, float]] = []
    key_values: dict[str, list[float]] = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        numeric_row: dict[str, float] = {}

        for key, value in row.items():
            if isinstance(value, (dict, list)):
                continue
            parsed = _to_float(value)
            if parsed is not None:
                numeric_row[key] = parsed

        test_parameters = row.get("TestParametersFlat")
        if isinstance(test_parameters, dict):
            for key, value in test_parameters.items():
                parsed = _to_float(value)
                if parsed is not None:
                    numeric_row[key] = parsed

        if not numeric_row:
            continue

        row_numeric_maps.append(numeric_row)
        for key, parsed in numeric_row.items():
            key_values.setdefault(key, []).append(parsed)

    return row_numeric_maps, key_values


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    clamped = min(1.0, max(0.0, q))
    pos = (len(ordered) - 1) * clamped
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))

    if lower == upper:
        return ordered[lower]

    weight = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _median(values: list[float]) -> float | None:
    return _percentile(values, 0.5)


def _score_key_for_term(term: str, key: str) -> float:
    term_compact = _normalize_compact(term)
    key_compact = _normalize_compact(key)

    score = 0.0
    if term_compact and term_compact == key_compact:
        score += 120.0
    if term_compact and term_compact in key_compact:
        score += 60.0
    if key_compact and key_compact in term_compact:
        score += 30.0

    term_words = set(_tokenize_words(term))
    key_words = set(_tokenize_words(key))
    overlap = term_words & key_words
    if overlap:
        score += 14.0 * len(overlap)
        score += 20.0 * (len(overlap) / max(1, len(term_words)))

    return score


def _expand_term_hints(term: str, question: str) -> list[str]:
    hints: list[str] = [term] if term.strip() else []
    combined = term.lower() if term.strip() else question.lower()

    for trigger, aliases in _TERM_TO_KEY_HINTS.items():
        if trigger in combined:
            hints.extend(aliases)

    if not hints:
        for token in _tokenize_words(question):
            if token not in _STOP_WORDS and len(token) >= 4:
                hints.append(token)

    # Keep order deterministic and remove duplicates.
    return list(dict.fromkeys(item.strip() for item in hints if item and item.strip()))


def _family_for_text(text: str) -> str | None:
    words = set(_tokenize_words(text))
    if words & {"force", "strength", "yield", "stress"}:
        return "force"
    if words & {"thickness", "wall", "specimen"}:
        return "thickness"
    if words & {"diameter"}:
        return "diameter"
    if words & {"extension", "strain", "elongation"}:
        return "extension"
    return None


def _key_matches_family(key: str, family: str | None) -> bool:
    if not family:
        return False
    key_family = _family_for_text(key)
    return key_family == family


def _match_key_for_term(
    term: str,
    question: str,
    key_values: dict[str, list[float]],
    prefer_non_limit: bool,
    excluded_keys: set[str] | None = None,
) -> str | None:
    if not key_values:
        return None

    excluded = excluded_keys or set()
    term_hints = _expand_term_hints(term, question)

    ranked: list[tuple[float, int, str]] = []
    for key, values in key_values.items():
        if key in excluded:
            continue

        direct_term_score = _score_key_for_term(term, key) if term.strip() else 0.0
        hint_score = max((_score_key_for_term(hint, key) for hint in term_hints), default=0.0)
        count_bonus = min(8.0, len(values) / 20.0)
        limit_penalty = 8.0 if prefer_non_limit and _is_limit_key(key) else 0.0
        # Prioritize the exact user term over broader aliases when scores are close.
        final_score = hint_score + (0.35 * direct_term_score) + count_bonus - limit_penalty
        ranked.append((final_score, len(values), key))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_score, _, best_key = ranked[0]
    if best_score > 0:
        return best_key

    fallback_pool = [item for item in ranked if not _is_limit_key(item[2])] if prefer_non_limit else ranked
    if not fallback_pool:
        fallback_pool = ranked

    fallback_pool.sort(key=lambda item: (-item[1], item[2]))
    return fallback_pool[0][2] if fallback_pool else None


def _extract_compliance_metric_term(question: str) -> str:
    q = question.strip().lower()

    patterns = [
        r"\b(?:is|are)\s+(?:the\s+)?(.+?)\s+(?:within|against|as\s+per|according\s+to|compliant)\b",
        r"\b(?:check|validate|assess)\s+(.+?)\s+(?:within|against|as\s+per|according\s+to)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
        if candidate:
            return candidate

    if "tensile strength" in q:
        return "tensile strength"
    if "maximum force" in q:
        return "maximum force"
    if "wall thickness" in q:
        return "wall thickness"
    if "thickness" in q:
        return "thickness"
    if "force" in q:
        return "force"
    return "values"


def _pick_limit_keys(
    question: str,
    metric_term: str,
    measurement_key: str,
    key_values: dict[str, list[float]],
) -> tuple[str | None, str | None]:
    candidate_keys = [key for key in key_values.keys() if key != measurement_key and _is_limit_key(key)]
    if not candidate_keys:
        return None, None

    family = set(_tokenize_words(question)) | set(_tokenize_words(metric_term)) | set(_tokenize_words(measurement_key))
    family = {word for word in family if word in _FAMILY_TOKENS}

    def family_bonus(key: str) -> float:
        if not family:
            return 0.0
        words = set(_tokenize_words(key))
        return 40.0 if words & family else 0.0

    def upper_score(key: str) -> float:
        words = set(_tokenize_words(key))
        return (
            family_bonus(key)
            + 16.0 * len(words & _UPPER_LIMIT_TOKENS)
            - 8.0 * len(words & {"shutdown"})
            + min(6.0, len(key_values.get(key, [])) / 30.0)
        )

    def lower_score(key: str) -> float:
        words = set(_tokenize_words(key))
        return (
            family_bonus(key)
            + 16.0 * len(words & _LOWER_LIMIT_TOKENS)
            - 8.0 * len(words & {"upper"})
            + min(6.0, len(key_values.get(key, [])) / 30.0)
        )

    upper_ranked = sorted(((upper_score(key), key) for key in candidate_keys), key=lambda item: (-item[0], item[1]))
    lower_ranked = sorted(((lower_score(key), key) for key in candidate_keys), key=lambda item: (-item[0], item[1]))

    upper_key = upper_ranked[0][1] if upper_ranked and upper_ranked[0][0] > 0 else None
    lower_key = lower_ranked[0][1] if lower_ranked and lower_ranked[0][0] > 0 else None

    if upper_key and lower_key and upper_key == lower_key:
        for _, candidate in lower_ranked[1:]:
            if candidate != upper_key:
                lower_key = candidate
                break
        else:
            lower_key = None

    return upper_key, lower_key


def _format_bound(value: float | None) -> str:
    if value is None:
        return "not set"
    return f"{value:.6g}"


def _build_generic_plausibility_insight(
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    key_values: dict[str, list[float]],
) -> InsightResponse | None:
    candidate_keys = []
    for key, values in key_values.items():
        if _is_limit_key(key):
            continue
        if len(values) < 10:
            continue
        words = set(_tokenize_words(key))
        family_bonus = 30.0 if words & _FAMILY_TOKENS else 0.0
        p10 = _percentile(values, 0.10)
        p90 = _percentile(values, 0.90)
        spread = (p90 - p10) if p10 is not None and p90 is not None else 0.0
        spread_score = math.log10(abs(spread) + 1.0)
        candidate_keys.append((family_bonus + spread_score, len(values), key))

    if not candidate_keys:
        return None

    candidate_keys.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected_keys = [key for _, _, key in candidate_keys[:3]]

    metric_notes: list[str] = []
    total_checks = 0
    total_violations = 0
    for key in selected_keys:
        values = key_values[key]
        p10 = _percentile(values, 0.10)
        p90 = _percentile(values, 0.90)
        if p10 is None or p90 is None:
            continue

        in_band = sum(1 for value in values if p10 <= value <= p90)
        out_of_band = len(values) - in_band
        total_checks += len(values)
        total_violations += out_of_band
        metric_notes.append(
            f"{key}: band [{p10:.6g}, {p90:.6g}], outside={out_of_band}/{len(values)}"
        )

    if not metric_notes:
        return None

    violation_rate = (100.0 * total_violations / total_checks) if total_checks else 0.0

    return InsightResponse(
        summary_3_sentences=[
            f"No explicit metric was named, so plausibility was evaluated across {len(metric_notes)} representative numeric parameters.",
            f"Empirical p10-p90 bands found {total_violations}/{total_checks} out-of-band values ({violation_rate:.1f}%) across selected parameters.",
            f"Parameter breakdown: {'; '.join(metric_notes)}.",
        ],
        anomaly_notes=[
            "This plausibility check uses empirical bands because standard-specific numeric thresholds were not provided in the query result."
        ],
        recommendation="Specify the target metric (for example maximum force or wall thickness) to run an explicit limit-based compliance check.",
        follow_up_questions=[
            "Run the same plausibility check for maximum force only.",
            "Show out-of-band rows grouped by tester.",
            "Compare these plausibility rates against another standard.",
        ],
        chart_config={"type": "bar", "x": "parameter", "y": "outside_count", "title": "Plausibility by parameter"},
        audit_log=[
            "Compliance insight generated using generic plausibility profile.",
            f"Input row count: {len(rows)}",
            f"Selected keys: {', '.join(selected_keys)}",
            f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
        ],
    )


def _build_validation_compliance_insight(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse | None:
    row_count = len(rows)
    question = str(getattr(plan, "user_question", ""))
    row_numeric_maps, key_values = _extract_numeric_key_values(rows)

    if not row_numeric_maps or not key_values:
        return InsightResponse(
            summary_3_sentences=[
                f"Compliance intent detected but {row_count} rows did not expose numeric fields for threshold checks.",
                "A deterministic pass/fail decision could not be computed from the returned payload.",
                "Expand the query to include numeric result columns or parameter values tied to explicit limits.",
            ],
            anomaly_notes=["No compliance anomaly computed because no numeric metrics were available."],
            recommendation="Return numeric metrics and their limits in the query output, then rerun compliance.",
            follow_up_questions=_follow_up_for_intent(IntentCategory.validation_compliance),
            chart_config={"type": "table", "title": "Compliance check unavailable"},
            audit_log=[
                "Compliance insight generated from deterministic fallback.",
                f"Input row count: {row_count}",
                "Numeric fields detected: 0",
                f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
            ],
        )

    metric_term = _extract_compliance_metric_term(question)
    metric_words = [word for word in _tokenize_words(metric_term) if word not in _STOP_WORDS]
    ambiguous_metric = metric_term in {"value", "values"} or "plausible" in metric_term or not metric_words
    if ambiguous_metric:
        generic = _build_generic_plausibility_insight(rows, stats, key_values)
        if generic is not None:
            return generic

    measurement_key = _match_key_for_term(metric_term, question, key_values, prefer_non_limit=True)
    if not measurement_key:
        measurement_key = _match_key_for_term(metric_term, question, key_values, prefer_non_limit=False)
    if not measurement_key:
        return None

    measurement_values = key_values.get(measurement_key, [])
    if not measurement_values:
        return None

    upper_limit_key, lower_limit_key = _pick_limit_keys(question, metric_term, measurement_key, key_values)
    upper_limit = _median(key_values[upper_limit_key]) if upper_limit_key else None
    lower_limit = _median(key_values[lower_limit_key]) if lower_limit_key else None

    bounds_source = "explicit limit fields"
    if upper_limit is None and lower_limit is None:
        lower_limit = _percentile(measurement_values, 0.10)
        upper_limit = _percentile(measurement_values, 0.90)
        bounds_source = "empirical 10-90% plausibility range"

    if lower_limit is not None and upper_limit is not None and lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit

    pass_count = 0
    fail_count = 0
    max_deviation = 0.0

    for value in measurement_values:
        lower_ok = lower_limit is None or value >= lower_limit
        upper_ok = upper_limit is None or value <= upper_limit
        if lower_ok and upper_ok:
            pass_count += 1
            continue

        fail_count += 1
        if lower_limit is not None and value < lower_limit:
            max_deviation = max(max_deviation, lower_limit - value)
        if upper_limit is not None and value > upper_limit:
            max_deviation = max(max_deviation, value - upper_limit)

    total = len(measurement_values)
    pass_pct = (100.0 * pass_count / total) if total else 0.0
    fail_pct = (100.0 * fail_count / total) if total else 0.0
    observed_min = min(measurement_values)
    observed_max = max(measurement_values)
    observed_median = _median(measurement_values) or 0.0

    metric_explicitly_named = metric_term not in {"value", "values"}
    if metric_explicitly_named:
        lead_sentence = f"Compliance check evaluated '{measurement_key}' for {total} matched records."
    else:
        lead_sentence = (
            f"No explicit metric was named; compliance check used '{measurement_key}' as the primary numeric signal "
            f"for {total} matched records."
        )

    summary = [
        lead_sentence,
        (
            f"Limits were {_format_bound(lower_limit)} to {_format_bound(upper_limit)} ({bounds_source}); "
            f"pass={pass_count} ({pass_pct:.1f}%), violations={fail_count} ({fail_pct:.1f}%)."
        ),
        (
            f"Observed '{measurement_key}' range is {observed_min:.6g} to {observed_max:.6g} "
            f"(median {observed_median:.6g})."
        ),
    ]

    anomaly_notes = []
    if fail_count > 0:
        anomaly_notes.append(
            f"Detected {fail_count} compliance violations; maximum deviation from bounds is {max_deviation:.6g}."
        )
    else:
        anomaly_notes.append("No compliance violations were detected in the evaluated records.")

    recommendation = (
        "Investigate violating records by tester, machine, and lot to confirm whether failures are process-related or limit-definition issues."
        if fail_count > 0
        else "Current subset stays within evaluated bounds; verify the same check on a wider time window for stability."
    )

    return InsightResponse(
        summary_3_sentences=summary,
        anomaly_notes=anomaly_notes,
        recommendation=recommendation,
        follow_up_questions=[
            "Show the violating records only, grouped by tester.",
            "Break compliance down by machine and standard.",
            "Compare this compliance profile with another material batch.",
        ],
        chart_config={
            "type": "bar",
            "x": "status",
            "y": "count",
            "title": f"Compliance for {measurement_key}",
        },
        audit_log=[
            "Compliance insight generated from deterministic numeric analysis.",
            f"Input row count: {row_count}",
            f"Measurement key: {measurement_key}",
            f"Upper limit key: {upper_limit_key or 'none'}",
            f"Lower limit key: {lower_limit_key or 'none'}",
            f"Bounds source: {bounds_source}",
            f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
        ],
    )


def _extract_hypothesis_terms(question: str) -> tuple[str, str]:
    q = question.strip()
    patterns = [
        r"\bchange\s+in\s+(.+?)\s+(?:influence|influences|affect|affects|impact|impacts)\s+(?:the\s+)?(.+?)(?:\s+for\b|\s+in\b|\s+with\b|[\?\.!]|$)",
        r"\bhow\s+does\s+(.+?)\s+(?:influence|affect|impact)\s+(?:the\s+)?(.+?)(?:\s+for\b|\s+in\b|\s+with\b|[\?\.!]|$)",
        r"\b(?:effect|impact)\s+of\s+(.+?)\s+on\s+(.+?)(?:\s+for\b|\s+in\b|\s+with\b|[\?\.!]|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if not match:
            continue
        x_term = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?:;")
        y_term = re.sub(r"\s+", " ", match.group(2)).strip(" .,!?:;")
        if x_term and y_term:
            return x_term, y_term

    lowered = q.lower()
    x_term = "wall thickness" if "wall thickness" in lowered else "thickness"
    if "maximum force" in lowered:
        y_term = "maximum force"
    elif "tensile strength" in lowered:
        y_term = "tensile strength"
    else:
        y_term = "force"

    return x_term, y_term


def _paired_numeric_values(row_numeric_maps: list[dict[str, float]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for numeric_row in row_numeric_maps:
        x = numeric_row.get(x_key)
        y = numeric_row.get(y_key)
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values


def _pearson_and_slope(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0, 0.0

    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    var_x = sum((value - mean_x) ** 2 for value in x_values)
    var_y = sum((value - mean_y) ** 2 for value in y_values)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=False))

    if var_x <= 0 or var_y <= 0:
        return 0.0, 0.0

    slope = covariance / var_x
    correlation = covariance / math.sqrt(var_x * var_y)
    return correlation, slope


def _build_hypothesis_insight(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse | None:
    row_count = len(rows)
    question = str(getattr(plan, "user_question", ""))
    row_numeric_maps, key_values = _extract_numeric_key_values(rows)

    if not row_numeric_maps or not key_values:
        return InsightResponse(
            summary_3_sentences=[
                f"Hypothesis intent detected but {row_count} rows did not contain paired numeric variables.",
                "Association statistics could not be computed from the current payload.",
                "Return explicit numeric x/y parameters in the query output and rerun the hypothesis probe.",
            ],
            anomaly_notes=["No hypothesis metric computed because paired numeric data was unavailable."],
            recommendation="Include numeric predictors and outcomes in the query projection before rerunning hypothesis analysis.",
            follow_up_questions=[
                "Return wall thickness and force metrics together for each test.",
                "Split the same hypothesis by machine type.",
                "Add a time window and compare whether the relationship changes.",
            ],
            chart_config={"type": "table", "title": "Hypothesis probe unavailable"},
            audit_log=[
                "Hypothesis insight generated from deterministic fallback.",
                f"Input row count: {row_count}",
                "Numeric fields detected: 0",
                f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
            ],
        )

    x_term, y_term = _extract_hypothesis_terms(question)
    x_key = _match_key_for_term(x_term, question, key_values, prefer_non_limit=True)
    y_key = _match_key_for_term(y_term, question, key_values, prefer_non_limit=False, excluded_keys={x_key} if x_key else None)

    if not x_key or not y_key:
        return None

    x_family = _family_for_text(x_term)
    y_family = _family_for_text(y_term)
    if x_family and y_family and x_family != y_family:
        if _key_matches_family(x_key, y_family) and _key_matches_family(y_key, x_family):
            x_key, y_key = y_key, x_key

    x_values, y_values = _paired_numeric_values(row_numeric_maps, x_key, y_key)
    if len(x_values) < 5:
        return InsightResponse(
            summary_3_sentences=[
                f"Hypothesis probe found only {len(x_values)} paired observations for '{x_key}' vs '{y_key}'.",
                "That sample is too small for stable association estimates.",
                "Broaden filters or include more records before drawing conclusions.",
            ],
            anomaly_notes=["Insufficient paired sample size for robust hypothesis statistics."],
            recommendation="Increase sample size and re-run with the same x/y variables to verify the effect direction.",
            follow_up_questions=[
                "Expand to a larger time range while keeping material fixed.",
                "Compare the same hypothesis for another material.",
                "Add tester-level grouping to check for confounding.",
            ],
            chart_config={"type": "scatter", "x": x_key, "y": y_key, "title": f"{y_key} vs {x_key}"},
            audit_log=[
                "Hypothesis insight generated from deterministic numeric analysis.",
                f"Input row count: {row_count}",
                f"X key: {x_key}",
                f"Y key: {y_key}",
                f"Paired count: {len(x_values)}",
            ],
        )

    correlation, slope = _pearson_and_slope(x_values, y_values)

    x_median = _median(x_values) or 0.0
    low_group = [y for x, y in zip(x_values, y_values, strict=False) if x <= x_median]
    high_group = [y for x, y in zip(x_values, y_values, strict=False) if x > x_median]

    if not low_group or not high_group:
        q25 = _percentile(x_values, 0.25)
        q75 = _percentile(x_values, 0.75)
        if q25 is not None and q75 is not None:
            low_group = [y for x, y in zip(x_values, y_values, strict=False) if x <= q25]
            high_group = [y for x, y in zip(x_values, y_values, strict=False) if x >= q75]

    group_comparison_sentence: str
    if low_group and high_group:
        low_mean = sum(low_group) / len(low_group)
        high_mean = sum(high_group) / len(high_group)
        delta_mean = high_mean - low_mean
        group_comparison_sentence = (
            f"high-{x_key} mean {y_key}={high_mean:.6g} vs low-{x_key} mean {y_key}={low_mean:.6g} "
            f"(delta {delta_mean:.6g})."
        )
    else:
        group_comparison_sentence = (
            f"{x_key} has limited spread in this subset, so high/low group contrast is not stable."
        )

    strength = "very weak"
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.2:
        strength = "weak"

    direction = "positive" if correlation > 0.05 else "negative" if correlation < -0.05 else "near-zero"
    relationship_sentence = (
        f"This indicates a {strength} {direction} association in this filtered subset; association alone does not prove causation."
    )

    recommendation = (
        "Control for test program, machine, and tester to confirm whether this relationship remains after removing confounders."
        if abs_corr >= 0.3
        else "The observed association is weak; segment by program/material and collect more data before acting on this hypothesis."
    )

    return InsightResponse(
        summary_3_sentences=[
            f"Hypothesis probe paired {len(x_values)} records for '{x_key}' versus '{y_key}'.",
            (
                f"Pearson r={correlation:.3f}, slope={slope:.6g} ({y_key} per 1 unit of {x_key}); "
                f"{group_comparison_sentence}"
            ),
            relationship_sentence,
        ],
        anomaly_notes=[
            "No statistical anomaly detector was applied in this step; result focuses on directional association only."
        ],
        recommendation=recommendation,
        follow_up_questions=[
            "Repeat this hypothesis after fixing one machine type only.",
            "Split by tester to see whether the slope is consistent.",
            "Run the same analysis on another material grade.",
        ],
        chart_config={"type": "scatter", "x": x_key, "y": y_key, "title": f"{y_key} vs {x_key}"},
        audit_log=[
            "Hypothesis insight generated from deterministic numeric analysis.",
            f"Input row count: {row_count}",
            f"X key: {x_key}",
            f"Y key: {y_key}",
            f"Paired count: {len(x_values)}",
            f"Correlation: {correlation:.6g}",
            f"Slope: {slope:.6g}",
            f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
        ],
    )


def _build_insight_mock(plan: Any, rows: list[dict[str, Any]], stats: dict[str, Any]) -> InsightResponse:
    row_count = len(rows)
    anomalies = stats.get("anomalies", []) if isinstance(stats, dict) else []

    if plan.intent == IntentCategory.comparison and rows:
        tester_rows = [row for row in rows if isinstance(row, dict) and "tester" in row]
        if tester_rows:
            metric_counts: dict[str, int] = {}
            for row in tester_rows:
                for key, value in row.items():
                    if key == "tester":
                        continue
                    if isinstance(value, (int, float)):
                        metric_counts[key] = metric_counts.get(key, 0) + 1

            if not metric_counts:
                for row in tester_rows:
                    for key in row.keys():
                        if key == "tester":
                            continue
                        metric_counts[key] = metric_counts.get(key, 0) + 1

            if not metric_counts:
                metric_counts = {"value": len(tester_rows)}

            primary_metric = sorted(metric_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            metric_label = primary_metric.replace("_", " ")

            formatted_pairs: list[str] = []
            numeric_values: list[float] = []
            for row in tester_rows:
                tester_name = str(row.get("tester", "unknown")).strip() or "unknown"
                raw_value = row.get(primary_metric)
                try:
                    numeric_value = float(raw_value)
                    numeric_values.append(numeric_value)
                    formatted_pairs.append(f"{tester_name}: {numeric_value:.6g}")
                except Exception:  # noqa: BLE001
                    formatted_pairs.append(f"{tester_name}: {raw_value}")

            if len(formatted_pairs) > 8:
                formatted_pairs = formatted_pairs[:8]
                formatted_pairs.append("...")

            summary = [
                f"Compared {metric_label} for {len(tester_rows)} tester groups.",
                f"{metric_label.title()} values: {', '.join(formatted_pairs)}.",
                "These values are computed directly from matched test records for the requested comparison.",
            ]

            if len(numeric_values) >= 2:
                difference = max(numeric_values) - min(numeric_values)
                recommendation = (
                    f"The gap between the compared {metric_label} values is {difference:.6g}. "
                    "Validate whether this spread is expected for the same program/material conditions."
                )
            else:
                recommendation = (
                    "Only one tester group was matched. Add another tester filter for a direct comparison."
                )

            audit_log = [
                "Insight generated from query output plus statistics payload.",
                f"Input row count: {row_count}",
                f"Stats keys: {', '.join(sorted(stats.keys())) if isinstance(stats, dict) and stats else 'none'}",
                f"Detected comparison result shape: tester + {primary_metric}.",
            ]

            return InsightResponse(
                summary_3_sentences=summary,
                anomaly_notes=["No major anomaly signal detected in this response."],
                recommendation=recommendation,
                follow_up_questions=[
                    "Compare the same testers for another parameter (e.g., specimen thickness).",
                    "Restrict the comparison to one test program or material.",
                    "Add a time window to check if the gap is stable over time.",
                ],
                chart_config={
                    "type": "bar",
                    "x": "tester",
                    "y": primary_metric,
                    "title": f"{metric_label.title()} by Tester",
                },
                audit_log=audit_log,
            )

    if plan.intent == IntentCategory.validation_compliance:
        compliance_insight = _build_validation_compliance_insight(plan, rows, stats)
        if compliance_insight is not None:
            return compliance_insight

    if plan.intent == IntentCategory.hypothesis:
        hypothesis_insight = _build_hypothesis_insight(plan, rows, stats)
        if hypothesis_insight is not None:
            return hypothesis_insight

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
    provider = settings.llm_provider.lower()
    if provider not in {"anthropic", "openai"}:
        return None

    gateway = ClaudeGateway()
    if not gateway.is_ready():
        return None

    model_name = settings.openai_model_insight if provider == "openai" else settings.anthropic_model_insight

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
        model=model_name,
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

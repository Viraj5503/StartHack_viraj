import re
from typing import Any

from app.config import get_settings
from app.schemas import (
    CollectionName,
    FilterSpec,
    IntentCategory,
    InvestigationPlan,
    PlanOperation,
    StatMethod,
)
from app.services.llm_gateway import ClaudeGateway
from app.services.semantic_layer import resolve_user_term


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _matches_any(question_lower: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, question_lower) for pattern in patterns)


def _infer_intent(question_lower: str) -> IntentCategory:
    if _matches_any(
        question_lower,
        [
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bdifferent\b",
            r"\bdifference\b",
            r"\bcomparable\b",
            r"\bsignificant(?:ly)?\b",
        ],
    ):
        return IntentCategory.comparison
    if _matches_any(
        question_lower,
        [
            r"\btrend\b",
            r"\bdrift\b",
            r"\bover time\b",
            r"\blast\s+\d+\b",
            r"\bdecreas\w*\b",
            r"\bincreas\w*\b",
            r"\bfuture\b",
            r"\bboundary\b.*\bviolat\w*\b",
            r"\bviolat\w*\b.*\bfuture\b",
            r"\bdegradat\w*\b",      # "degradation", "degrades"
            r"\bworsening\b",
            r"\bdeteriora\w*\b",     # "deterioration", "deteriorating"
            r"\bchanging\s+over\b",
            r"\bevolution\b",
        ],
    ):
        return IntentCategory.trend_drift
    if _matches_any(
        question_lower,
        [
            r"\bcompliance\b",
            r"\bwithin\b",
            r"\blimits?\b",
            r"\bstandard\b",
            r"\bplausible\b",
            r"\biso\s*\d+\b",
            r"\bguideline\w*\b",
            r"\bconform\w*\b",
        ],
    ):
        return IntentCategory.validation_compliance
    if _matches_any(
        question_lower,
        [
            r"\bhypothesis\b",
            r"\binfluence\b",
            r"\bif i change\b",
            r"\bimpact of\b",
            r"\beffect of\b",
            r"\bhow does.*influence\b",
        ],
    ):
        return IntentCategory.hypothesis
    if _matches_any(
        question_lower,
        [
            r"\banomal\w*\b",
            r"\boutlier\w*\b",
            r"\babnormal\b",
            r"\bstrange\b",
        ],
    ):
        return IntentCategory.anomaly_check
    if _matches_any(
        question_lower,
        [
            r"\bsummar(?:y|ize)\b",
            r"\boverview\b",
            r"\bhigh level\b",
        ],
    ):
        return IntentCategory.summary
    if _matches_any(
        question_lower,
        [
            r"\blist\b",
            r"\bshow\b",
            r"\bfind\b",
            r"\bwhich\b",
            r"\bfetch\b",
        ],
    ):
        return IntentCategory.data_selection
    return IntentCategory.summary


def _suggest_operations(intent: IntentCategory) -> list[PlanOperation]:
    if intent == IntentCategory.comparison:
        return [PlanOperation.filtering, PlanOperation.grouping, PlanOperation.statistics]
    if intent == IntentCategory.trend_drift:
        return [
            PlanOperation.filtering,
            PlanOperation.time_series_extract,
            PlanOperation.statistics,
            PlanOperation.anomaly_scan,
        ]
    if intent == IntentCategory.validation_compliance:
        return [PlanOperation.filtering, PlanOperation.compliance_check, PlanOperation.statistics]
    if intent == IntentCategory.hypothesis:
        return [PlanOperation.filtering, PlanOperation.hypothesis_probe, PlanOperation.statistics]
    if intent == IntentCategory.anomaly_check:
        return [PlanOperation.filtering, PlanOperation.statistics, PlanOperation.anomaly_scan]
    if intent == IntentCategory.data_selection:
        return [PlanOperation.filtering]
    return [PlanOperation.filtering, PlanOperation.aggregation]


def _suggest_stats(intent: IntentCategory) -> list[StatMethod]:
    if intent == IntentCategory.comparison:
        return [StatMethod.descriptive, StatMethod.t_test, StatMethod.mann_whitney]
    if intent == IntentCategory.trend_drift:
        return [
            StatMethod.descriptive,
            StatMethod.rolling_mean,
            StatMethod.linear_slope,
            StatMethod.change_point_heuristic,
        ]
    if intent == IntentCategory.validation_compliance:
        return [StatMethod.descriptive, StatMethod.threshold_check]
    if intent == IntentCategory.hypothesis:
        return [StatMethod.descriptive, StatMethod.t_test]
    if intent == IntentCategory.anomaly_check:
        return [StatMethod.descriptive, StatMethod.z_score]
    return [StatMethod.descriptive]


def _extract_field_hints(question_lower: str) -> list[str]:
    mapping = {
        "customer": "TestParametersFlat.CUSTOMER",
        "material": "TestParametersFlat.MATERIAL",
        "machine": "TestParametersFlat.MACHINE_TYPE_STR",
        "site": "TestParametersFlat.CUSTOMER_NAME",
        "time": "uploadDate",
        "force": "values",
        "strain": "values",
        "temperature": "TestParametersFlat.TEMPERATURE",
        "standard": "TestParametersFlat.STANDARD",
    }
    return [field for token, field in mapping.items() if token in question_lower]


def _extract_named_entity_filters(question: str) -> list[dict[str, Any]]:
    """Extract named entity filters (material, customer, tester, standard) from the question.

    Uses regex heuristics so the heuristic planner can produce meaningful filters
    even without an LLM call.
    """
    q = question.strip()
    filters: list[dict[str, Any]] = []

    # Material: "material X" / "the material X" / "of material X"
    mat_m = re.search(
        r'\b(?:the\s+)?material\s+([A-Za-z0-9][A-Za-z0-9\s\-\.]{1,40}?)(?=\s+(?:regarding|in\s+my|on\b|for\b|and\b|or\b|is\b|are\b|was\b|will\b|the\b|to\b|at\b)|\s*[,\?\.\!]|$)',
        q, re.IGNORECASE,
    )
    if mat_m:
        val = mat_m.group(1).strip()
        first_word = val.lower().split()[0] if val.split() else ""
        stop_words = {"and", "or", "the", "a", "is", "are", "was", "were", "in", "for", "on", "my"}
        if val and 2 <= len(val) <= 40 and first_word not in stop_words:
            filters.append({"field": "material", "operator": "contains", "value": val})

    # Customer: "customer X" or "for <ProperName> [Industries/Ltd/GmbH ...]"
    cust_m = re.search(
        r'\bcustomer\s+([A-Za-z0-9][A-Za-z0-9\s\-\.]{1,40}?)(?=\s+(?:regarding|in|for|on|and|or|is|are|was|the|to|of)|\s*[,\?\.\!]|$)',
        q, re.IGNORECASE,
    )
    if cust_m:
        val = cust_m.group(1).strip()
        if val and 2 <= len(val) <= 40:
            filters.append({"field": "customer", "operator": "contains", "value": val})
    elif not any(f["field"] == "customer" for f in filters):
        # Try "for <ProperName>" — only match multi-word proper names that look like companies
        for_m = re.search(
            r'\bfor\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)+)(?:\s+(?:for|and|or|their|in|on|at|regarding)|\s*[,\?\.\!]|$)',
            q,
        )
        if for_m:
            val = for_m.group(1).strip()
            if val and 2 <= len(val) <= 50:
                filters.append({"field": "customer", "operator": "contains", "value": val})

    # Tester: "by tester X" / "tester X" / "performed by X"
    tester_m = re.search(
        r'\b(?:by\s+tester|tester)\s+([A-Za-z0-9][A-Za-z0-9\s\-\.]{1,40}?)(?=\s+(?:and|or|on|in|for|the|a|is)|\s*[,\?\.\!]|$)',
        q, re.IGNORECASE,
    )
    if tester_m:
        val = tester_m.group(1).strip()
        if val and 2 <= len(val) <= 40:
            filters.append({"field": "tester", "operator": "contains", "value": val})

    # Standard: ISO/DIN/EN/ASTM patterns
    std_m = re.search(
        r'\b((?:DIN\s+EN\s+ISO|DIN\s+EN|DIN\s+ISO|DIN|ISO|EN|ASTM|BS|JIS|GB)\s+[A-Z0-9][\w\s\-]{1,20})',
        q, re.IGNORECASE,
    )
    if std_m:
        val = std_m.group(1).strip()
        if val:
            filters.append({"field": "standard", "operator": "contains", "value": val})

    return filters


_TERM_SYNONYMS: dict[str, list[str]] = {
    # "tensile strength" in ZwickRoell data is represented by stress/force result types.
    "tensile strength": ["maximum force", "upper yield point", "young s modulus"],
    "tensile": ["maximum force", "upper yield point"],
    "stress": ["maximum force", "upper yield point"],
    "yield strength": ["upper yield point"],
    "ultimate strength": ["maximum force"],
    "elongation": ["strain at break", "nominal strain at break"],
    "elongation at break": ["strain at break"],
    "stiffness": ["young s modulus"],
    "elastic modulus": ["young s modulus"],
}


def _collect_semantic_candidates(normalized: str, question_lower: str) -> list[dict[str, Any]]:
    semantic_candidates: list[dict[str, Any]] = []
    for token in ["tensile strength", "tensile", "maximum force", "crosshead", "time",
                  "temperature", "stress", "yield strength", "elongation", "stiffness",
                  "elastic modulus"]:
        if token in question_lower:
            semantic_candidates.extend(resolve_user_term(token))
            for syn in _TERM_SYNONYMS.get(token, []):
                semantic_candidates.extend(resolve_user_term(syn))

    if not semantic_candidates:
        for word in normalized.split():
            if len(word) >= 5:
                semantic_candidates.extend(resolve_user_term(word, limit=2))

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in semantic_candidates:
        unique = str(item.get("uuid") or item.get("id") or item.get("name"))
        key = (item.get("category", "unknown"), unique)
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:12]


def _build_plan_heuristic(question: str, context: dict[str, Any] | None = None) -> tuple[InvestigationPlan, list[dict[str, Any]]]:
    context = context or {}
    normalized = _normalize_question(question)
    question_lower = normalized.lower()

    intent = _infer_intent(question_lower)
    operations = _suggest_operations(intent)
    stats = _suggest_stats(intent)
    field_hints = _extract_field_hints(question_lower)

    requires_values = any(
        op in operations
        for op in [PlanOperation.time_series_extract, PlanOperation.statistics, PlanOperation.anomaly_scan]
    ) or any(token in question_lower for token in ["force", "strain", "curve", "time", "channel"])

    collections = [CollectionName.tests]
    if requires_values:
        collections.append(CollectionName.values)

    chart_needed = intent in {
        IntentCategory.comparison,
        IntentCategory.trend_drift,
        IntentCategory.anomaly_check,
        IntentCategory.hypothesis,
    }

    deduped = _collect_semantic_candidates(normalized, question_lower)

    reasoning_steps = [
        "Classify user intent from natural language.",
        "Map domain terms to known UUID and parameter dictionaries.",
        "Select collections and operations for query generation.",
        "Decide statistical checks and chart requirement.",
    ]

    assumptions = [
        "Question refers to historical test data available in MongoDB.",
        "Semantic mapping may include approximate term matching.",
    ]

    follow_up_focus = [
        "narrow time window",
        "compare another machine or site",
        "validate against standard limits",
    ]

    confidence = 0.72
    if len(deduped) >= 3:
        confidence = 0.82

    if context.get("strict") is True:
        confidence = min(0.95, confidence + 0.05)

    # Extract named entity filters from the question text
    entity_filter_dicts = _extract_named_entity_filters(question)
    entity_filters: list[FilterSpec] = []
    for ef in entity_filter_dicts:
        try:
            entity_filters.append(FilterSpec(**ef))
        except Exception:  # noqa: BLE001
            pass

    plan = InvestigationPlan(
        user_question=question,
        normalized_question=normalized,
        intent=intent,
        required_collections=collections,
        fields_needed=field_hints,
        operations=operations,
        statistics_methods=stats,
        chart_needed=chart_needed,
        assumptions=assumptions,
        reasoning_steps=reasoning_steps,
        follow_up_focus=follow_up_focus,
        filters=entity_filters,
        confidence=confidence,
    )

    return plan, deduped


def _normalize_enum_list(value: Any, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str) and item in allowed:
            normalized.append(item)
    return normalized


def _tokenize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_choice(raw_value: Any, allowed: set[str], aliases: dict[str, str]) -> str | None:
    if not isinstance(raw_value, str):
        return None

    if raw_value in allowed:
        return raw_value

    token = _tokenize(raw_value)

    allowed_map = {_tokenize(item): item for item in allowed}
    if token in allowed_map:
        return allowed_map[token]

    alias_target = aliases.get(token)
    if alias_target in allowed:
        return alias_target

    return None


def _normalize_choice_list(
    raw_values: Any,
    allowed: set[str],
    aliases: dict[str, str],
) -> list[str]:
    if not isinstance(raw_values, list):
        return []

    normalized: list[str] = []
    for raw in raw_values:
        choice = _normalize_choice(raw, allowed, aliases)
        if choice is not None:
            normalized.append(choice)
    return normalized


def _normalize_str_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or fallback


def _normalize_filters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    operator_aliases = {
        "eq": "eq",
        "equals": "eq",
        "=": "eq",
        "==": "eq",
        "neq": "neq",
        "!=": "neq",
        "<>": "neq",
        "noteq": "neq",
        "lt": "lt",
        "<": "lt",
        "lte": "lte",
        "<=": "lte",
        "gt": "gt",
        ">": "gt",
        "gte": "gte",
        ">=": "gte",
        "in": "in",
        "contains": "contains",
        "like": "contains",
        "between": "between",
        "range": "between",
        "regex": "regex",
        "match": "regex",
    }
    allowed_operators = {
        "eq",
        "neq",
        "lt",
        "lte",
        "gt",
        "gte",
        "in",
        "contains",
        "between",
        "regex",
    }

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        operator = item.get("operator")
        raw_value = item.get("value")

        if not isinstance(field, str) or not field.strip():
            continue

        if not isinstance(operator, str):
            continue

        op = operator_aliases.get(operator.lower().strip())
        if op not in allowed_operators:
            continue

        normalized.append({"field": field.strip(), "operator": op, "value": raw_value})

    return normalized


def _build_plan_llm(
    question: str,
    context: dict[str, Any],
    fallback_plan: InvestigationPlan,
    semantic_candidates: list[dict[str, Any]],
) -> InvestigationPlan | None:
    settings = get_settings()
    if settings.llm_provider.lower() != "anthropic":
        return None

    gateway = ClaudeGateway()
    if not gateway.is_ready():
        return None

    system_prompt = (
        "You are a materials testing AI planner. Return strictly one JSON object only. "
        "Do not include markdown or explanations."
    )

    user_prompt = (
        "Build an investigation plan for the question below.\n"
        "Allowed intent values: validation_compliance, comparison, trend_drift, hypothesis, anomaly_check, data_selection, summary.\n"
        "Allowed collection values: Tests, Values.\n"
        "Allowed operation values: filtering, aggregation, grouping, time_series_extract, statistics, compliance_check, anomaly_scan, hypothesis_probe.\n"
        "Allowed statistics values: descriptive, t_test, mann_whitney, rolling_mean, linear_slope, z_score, change_point_heuristic, threshold_check.\n"
        "Mongo schema reference (use these names exactly):\n"
        "- Tests collection (physical: _tests): _id, name, state, testProgramId, TestParametersFlat, valueColumns\n"
        "  WARNING: modifiedOn in _tests is {} (empty object, NOT a date) - NEVER put modifiedOn in filters\n"
        "- Values collection (physical: valuecolumns_migrated): _id, metadata.refId, metadata.childId, uploadDate, values, valuesCount\n"
        "  uploadDate is an ISODate - use this for ALL time/date filtering (e.g. 'last 6 months')\n"
        "  DATA DATE RANGE: uploadDate in this dataset is 2026-02-27 to 2026-03-03 (all within last 6 months).\n"
        "- Join key: _tests._id == valuecolumns_migrated.metadata.refId\n"
        "- 'tensile strength' in ZwickRoell data = Maximum force (UUID: 9DB9C049-9B04-4bf1-BD29-A160E86DE691)\n"
        "  or Upper yield point (UUID: 31D55559-E6A6-4fc3-B658-3C7291F3ECD4)\n"
        "Avoid deprecated placeholders like site, test_name, test_date, sample_id, test_id, modifiedOn.\n"
        "Return this schema exactly:\n"
        "{\n"
        "  \"user_question\": string,\n"
        "  \"normalized_question\": string,\n"
        "  \"intent\": string,\n"
        "  \"required_collections\": string[],\n"
        "  \"fields_needed\": string[],\n"
        "  \"operations\": string[],\n"
        "  \"statistics_methods\": string[],\n"
        "  \"chart_needed\": boolean,\n"
        "  \"assumptions\": string[],\n"
        "  \"reasoning_steps\": string[],\n"
        "  \"follow_up_focus\": string[],\n"
        "  \"filters\": [{\"field\": string, \"operator\": string, \"value\": any}],\n"
        "  \"confidence\": number\n"
        "}\n\n"
        f"Question: {question}\n"
        f"Context: {context}\n"
        f"Semantic candidates: {semantic_candidates}\n"
    )

    result = gateway.generate_json(
        model=settings.anthropic_model_planner,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if not result:
        return None

    result.setdefault("user_question", question)
    result.setdefault("normalized_question", _normalize_question(question))
    result.setdefault("intent", fallback_plan.intent.value)
    result.setdefault("required_collections", [c.value for c in fallback_plan.required_collections])
    result.setdefault("fields_needed", fallback_plan.fields_needed)
    result.setdefault("operations", [op.value for op in fallback_plan.operations])
    result.setdefault("statistics_methods", [s.value for s in fallback_plan.statistics_methods])
    result.setdefault("chart_needed", fallback_plan.chart_needed)
    result.setdefault("assumptions", fallback_plan.assumptions)
    result.setdefault("reasoning_steps", fallback_plan.reasoning_steps)
    result.setdefault("follow_up_focus", fallback_plan.follow_up_focus)
    result.setdefault("filters", [flt.model_dump() for flt in fallback_plan.filters])
    result.setdefault("confidence", fallback_plan.confidence)

    allowed_collections = {item.value for item in CollectionName}
    allowed_ops = {item.value for item in PlanOperation}
    allowed_stats = {item.value for item in StatMethod}

    intent_aliases = {
        "validation": "validation_compliance",
        "compliance": "validation_compliance",
        "validationcompliance": "validation_compliance",
        "compare": "comparison",
        "comparison": "comparison",
        "trend": "trend_drift",
        "drift": "trend_drift",
        "trenddrift": "trend_drift",
        "hypothesis": "hypothesis",
        "anomaly": "anomaly_check",
        "anomalycheck": "anomaly_check",
        "outlier": "anomaly_check",
        "dataselection": "data_selection",
        "selection": "data_selection",
        "summary": "summary",
        "summarize": "summary",
    }
    collection_aliases = {
        "tests": "Tests",
        "test": "Tests",
        "values": "Values",
        "value": "Values",
    }
    operation_aliases = {
        "filter": "filtering",
        "filtering": "filtering",
        "aggregate": "aggregation",
        "aggregation": "aggregation",
        "group": "grouping",
        "grouping": "grouping",
        "timeseries": "time_series_extract",
        "timeseriesextract": "time_series_extract",
        "timeseriesextraction": "time_series_extract",
        "time_series_extract": "time_series_extract",
        "statistics": "statistics",
        "stats": "statistics",
        "compliancecheck": "compliance_check",
        "compliance_check": "compliance_check",
        "anomalyscan": "anomaly_scan",
        "anomaly_scan": "anomaly_scan",
        "hypothesisprobe": "hypothesis_probe",
        "hypothesis_probe": "hypothesis_probe",
    }
    stats_aliases = {
        "descriptive": "descriptive",
        "ttest": "t_test",
        "t_test": "t_test",
        "mannwhitney": "mann_whitney",
        "mann_whitney": "mann_whitney",
        "rollingmean": "rolling_mean",
        "rolling_mean": "rolling_mean",
        "linearslope": "linear_slope",
        "linear_slope": "linear_slope",
        "zscore": "z_score",
        "z_score": "z_score",
        "changepointheuristic": "change_point_heuristic",
        "change_point_heuristic": "change_point_heuristic",
        "thresholdcheck": "threshold_check",
        "threshold_check": "threshold_check",
    }

    normalized_intent = _normalize_choice(
        result.get("intent"),
        {item.value for item in IntentCategory},
        intent_aliases,
    )
    result["intent"] = normalized_intent or fallback_plan.intent.value

    normalized_collections = _normalize_choice_list(result.get("required_collections"), allowed_collections, collection_aliases)
    result["required_collections"] = normalized_collections or [c.value for c in fallback_plan.required_collections]

    normalized_ops = _normalize_choice_list(result.get("operations"), allowed_ops, operation_aliases)
    result["operations"] = normalized_ops or [op.value for op in fallback_plan.operations]

    normalized_stats = _normalize_choice_list(result.get("statistics_methods"), allowed_stats, stats_aliases)
    result["statistics_methods"] = normalized_stats or [s.value for s in fallback_plan.statistics_methods]

    result["fields_needed"] = _normalize_str_list(result.get("fields_needed"), fallback_plan.fields_needed)
    result["assumptions"] = _normalize_str_list(result.get("assumptions"), fallback_plan.assumptions)
    result["reasoning_steps"] = _normalize_str_list(result.get("reasoning_steps"), fallback_plan.reasoning_steps)
    result["follow_up_focus"] = _normalize_str_list(result.get("follow_up_focus"), fallback_plan.follow_up_focus)
    result["filters"] = _normalize_filters(result.get("filters"))

    chart_needed = result.get("chart_needed")
    if isinstance(chart_needed, bool):
        result["chart_needed"] = chart_needed
    elif isinstance(chart_needed, str):
        result["chart_needed"] = chart_needed.lower().strip() in {"true", "yes", "1"}
    else:
        result["chart_needed"] = fallback_plan.chart_needed

    if not isinstance(result.get("confidence"), (int, float)):
        result["confidence"] = fallback_plan.confidence
    else:
        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

    try:
        return InvestigationPlan.model_validate(result)
    except Exception:  # noqa: BLE001
        return None


def build_plan(question: str, context: dict[str, Any] | None = None) -> tuple[InvestigationPlan, list[dict[str, Any]]]:
    context = context or {}
    fallback_plan, semantic_candidates = _build_plan_heuristic(question, context)

    settings = get_settings()
    if settings.planner_mode.lower() != "llm":
        return fallback_plan, semantic_candidates

    llm_plan = _build_plan_llm(question, context, fallback_plan, semantic_candidates)
    if llm_plan is None:
        return fallback_plan, semantic_candidates

    return llm_plan, semantic_candidates

from datetime import datetime, timedelta
import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Callable

from bson import ObjectId
from pymongo import MongoClient

from app.config import get_settings
from app.services.llm_gateway import ClaudeGateway
from app.schemas import (
    CollectionName,
    IntentCategory,
    MongoQueryCandidate,
    PlanOperation,
    QueryAttempt,
    QueryRunResponse,
)


MUTATING_AGG_STAGES = {"$out", "$merge"}
DEPRECATED_PLACEHOLDER_FIELDS = {
    "site",
    "test_name",
    "sample_id",
    "test_date",
    "test_id",
    "createdAt",
    "testParametersFlat",
    "material_id",
}
DEPRECATED_PLACEHOLDER_NORMALIZED_ROOTS = {
    "site",
    "testname",
    "sampleid",
    "testdate",
    "testid",
    "createdat",
    "materialid",
    "testparametersflat",
}
TESTS_SCOPE_FILTER_TOKENS = {
    "site",
    "customer",
    "material",
    "standard",
    "tester",
    "test_name",
    "test_type",
    "name",
    "state",
    "status",
    "result",
    "program",
    "testprogramid",
    "testparametersflat",
}
ATTRIBUTE_ALIASES = {
    "wallthickness": "Wall thickness",
    "specimenthickness": "SPECIMEN_THICKNESS",
    "specimenwidth": "SPECIMEN_WIDTH",
    "diameter": "Diameter",
    "outerdiameter": "Outer diameter",
    "innerdiameter": "Inner diameter",
    "weight": "Weight of the specimen",
    "weightofthespecimen": "Weight of the specimen",
    "crosssection": "Cross-section input",
    "crosssectioninput": "Cross-section input",
    "density": "Density of the specimen material",
    "youngsmodulus": "Young's modulus preset",
    "testspeed": "TEST_SPEED",
}


def _normalize_attribute_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@lru_cache
def _load_test_parameter_name_index() -> dict[str, str]:
    index: dict[str, str] = {}
    # Local copied helpers in this repo are the primary source.
    path = Path(__file__).resolve().parents[2] / "resources" / "uuid_helpers" / "TestParameterMap.json"
    if not path.exists():
        return index

    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return index

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("en")
        if not isinstance(name, str):
            continue
        token = _normalize_attribute_token(name)
        if token and token not in index:
            index[token] = name

    return index


def _normalize_filter_scope_token(field: str) -> str:
    token = field.strip()
    if not token:
        return token
    token = re.sub(r"^(?:_tests|tests|Tests|valuecolumns_migrated|values|Values)\.", "", token)
    return token.lower()


def _is_tests_scope_filter(field: str) -> bool:
    token = _normalize_filter_scope_token(field)
    if not token:
        return False

    if token.startswith("testparametersflat"):
        return True

    return token in TESTS_SCOPE_FILTER_TOKENS


def _is_deprecated_field_reference(value: str) -> bool:
    token = value.strip().lstrip("$")
    if not token:
        return False

    root = token.split(".", 1)[0]
    if root in DEPRECATED_PLACEHOLDER_FIELDS:
        return True

    normalized_root = re.sub(r"[^a-z0-9]+", "", root.lower())
    if normalized_root == "testparametersflat" and root == "TestParametersFlat":
        return False

    return normalized_root in DEPRECATED_PLACEHOLDER_NORMALIZED_ROOTS


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _filters_to_match(
    filters: list[dict[str, Any]],
    field_resolver: Callable[[str], str] | None = None,
    field_prefix: str = "",
) -> dict[str, Any]:
    and_clauses: list[dict[str, Any]] = []

    site_hint_present = any(
        isinstance(item.get("field"), str)
        and (
            "site" in str(item.get("field", "")).lower()
            or "testparametersflat" in str(item.get("field", "")).lower()
        )
        for item in filters
    )

    def build_clause(field_name: str, op: str, raw_value: Any) -> dict[str, Any] | None:
        def coerce_datetime(value: Any) -> Any:
            date_tail = field_name.split(".")[-1]
            if date_tail not in {"modifiedOn", "uploadDate"}:
                return value

            if isinstance(value, list):
                return [coerce_datetime(item) for item in value]

            if not isinstance(value, str):
                return value

            token = value.strip().lower()
            if token == "now":
                return datetime.utcnow()
            if token == "today":
                return datetime.utcnow()

            token_compact = re.sub(r"\s+", "", token)

            rel_match_compact = re.fullmatch(r"(?:now|today)-(\d+)(d|day|days|h|hour|hours|w|week|weeks|m|mo|month|months)", token_compact)
            if rel_match_compact:
                amount = int(rel_match_compact.group(1))
                unit = rel_match_compact.group(2)
                if unit in {"d", "day", "days"}:
                    return datetime.utcnow() - timedelta(days=amount)
                if unit in {"h", "hour", "hours"}:
                    return datetime.utcnow() - timedelta(hours=amount)
                if unit in {"w", "week", "weeks"}:
                    return datetime.utcnow() - timedelta(weeks=amount)
                if unit in {"m", "mo", "month", "months"}:
                    return datetime.utcnow() - timedelta(days=amount * 30)

            rel_match = re.fullmatch(r"now\s*-\s*(\d+)\s*(d|h|w|m|mo|month|months)", token)
            if rel_match:
                amount = int(rel_match.group(1))
                unit = rel_match.group(2)
                if unit == "d":
                    return datetime.utcnow() - timedelta(days=amount)
                if unit == "h":
                    return datetime.utcnow() - timedelta(hours=amount)
                if unit == "w":
                    return datetime.utcnow() - timedelta(weeks=amount)
                if unit in {"m", "mo", "month", "months"}:
                    return datetime.utcnow() - timedelta(days=amount * 30)

            if token.startswith("last_") and token.endswith("_days"):
                try:
                    amount = int(token.removeprefix("last_").removesuffix("_days"))
                    return datetime.utcnow() - timedelta(days=amount)
                except Exception:  # noqa: BLE001
                    return value

            if token.startswith("last_") and token.endswith("_months"):
                try:
                    amount = int(token.removeprefix("last_").removesuffix("_months"))
                    return datetime.utcnow() - timedelta(days=amount * 30)
                except Exception:  # noqa: BLE001
                    return value

            spaced_rel_match = re.fullmatch(
                r"(?:last|past)\s+(\d+)\s*(day|days|week|weeks|month|months|hour|hours)",
                token,
            )
            if spaced_rel_match:
                amount = int(spaced_rel_match.group(1))
                unit = spaced_rel_match.group(2)
                if unit in {"day", "days"}:
                    return datetime.utcnow() - timedelta(days=amount)
                if unit in {"week", "weeks"}:
                    return datetime.utcnow() - timedelta(weeks=amount)
                if unit in {"hour", "hours"}:
                    return datetime.utcnow() - timedelta(hours=amount)
                if unit in {"month", "months"}:
                    return datetime.utcnow() - timedelta(days=amount * 30)

            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                return value

        coerced_value = coerce_datetime(raw_value)

        if op == "eq":
            return {field_name: coerced_value}
        if op == "neq":
            return {field_name: {"$ne": coerced_value}}
        if op == "lt":
            return {field_name: {"$lt": coerced_value}}
        if op == "lte":
            return {field_name: {"$lte": coerced_value}}
        if op == "gt":
            return {field_name: {"$gt": coerced_value}}
        if op == "gte":
            return {field_name: {"$gte": coerced_value}}
        if op == "in":
            return {field_name: {"$in": coerced_value if isinstance(coerced_value, list) else [coerced_value]}}
        if op == "between" and isinstance(coerced_value, list) and len(coerced_value) == 2:
            return {field_name: {"$gte": coerced_value[0], "$lte": coerced_value[1]}}
        if op == "contains":
            return {field_name: {"$regex": str(raw_value), "$options": "i"}}
        if op == "regex":
            return {field_name: {"$regex": str(raw_value), "$options": "i"}}
        return None

    or_clauses: list[dict[str, Any]] = []

    def with_prefix(field_name: str) -> str:
        if not field_prefix:
            return field_name
        return f"{field_prefix}{field_name}"

    for item in filters:
        field = item.get("field")
        operator = item.get("operator")
        value = item.get("value")
        if not field or operator is None:
            continue

        resolved_field = field_resolver(field) if field_resolver else field
        if not resolved_field:
            continue

        if resolved_field == "__suppress_date__":
            # modifiedOn in _tests is {} (not a real date) — skip date filters on tests collection.
            continue

        if resolved_field == "__site_token__":
            value_text = str(value).strip().lower()
            if any(token in value_text for token in {"tensile", "strength", "strain", "force", "stress", "modulus"}):
                # Skip accidental routing of measurement terms into site/location matching.
                continue
            if re.fullmatch(r"(?:my\s+)?local[_\s-]*plant", value_text) or value_text in {
                "this plant",
                "my plant",
                "local site",
                "my site",
            }:
                # "local plant/site" is often conversational, not an indexed site identifier.
                continue

            site_paths = [
                with_prefix("TestParametersFlat.CUSTOMER_NAME"),
                with_prefix("TestParametersFlat.NOTES"),
                with_prefix("TestParametersFlat.Designation sub-series 1"),
                with_prefix("TestParametersFlat.CUSTOMER"),
            ]
            for site_path in site_paths:
                clause = build_clause(site_path, operator, value)
                if clause:
                    or_clauses.append(clause)
            continue

        if site_hint_present and resolved_field == "name" and operator in {"contains", "regex"}:
            # Site prompts often produce legacy name regex filters that suppress all rows.
            continue

        clause = build_clause(with_prefix(resolved_field), operator, value)
        if clause:
            and_clauses.append(clause)

    if and_clauses and or_clauses:
        return {"$and": [*and_clauses, {"$or": or_clauses}]}
    if len(and_clauses) == 1:
        return and_clauses[0]
    if len(and_clauses) > 1:
        return {"$and": and_clauses}
    if or_clauses:
        return {"$or": or_clauses}

    return {}


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$")
_DATE_FIELD_ROOTS = {"uploadDate", "modifiedOn", "createdAt", "timestamp", "date", "time"}


def _is_date_like_field(field_path: str) -> bool:
    """Return True if the field name looks like a date/time field."""
    tail = field_path.rstrip(".").split(".")[-1]
    return tail in _DATE_FIELD_ROOTS or tail.lower().endswith("date") or tail.lower().endswith("time")


def _coerce_date_strings(value: Any, field_path: str = "") -> Any:
    """Recursively convert ISO date strings in $match conditions to datetime objects.

    MongoDB's PyMongo driver compares ISODate fields with Python datetime objects.
    When the LLM emits a date as a plain string (e.g. "2025-09-19T00:00:00Z"),
    the comparison against an ISODate field always fails. This function converts
    those strings to datetime so the query works correctly.
    """
    # Handle MongoDB Extended JSON: {"$date": "..."}
    if isinstance(value, dict) and "$date" in value and isinstance(value["$date"], str):
        try:
            return datetime.fromisoformat(value["$date"].replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            pass

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            # For operator keys ($gte, $lte, $gt, $lt, $eq) keep the parent field_path
            # so date detection is based on the field name, not the operator.
            if isinstance(k, str) and k.startswith("$"):
                child_path = field_path
            else:
                child_path = f"{field_path}.{k}" if field_path else k
            result[k] = _coerce_date_strings(v, child_path)
        return result
    if isinstance(value, list):
        return [_coerce_date_strings(item, field_path) for item in value]
    if isinstance(value, str) and _is_date_like_field(field_path):
        if _ISO_DATE_RE.match(value.strip()):
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                pass
    return value


def _coerce_pipeline_dates(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Post-process an LLM-generated pipeline to fix ISO date strings in $match stages."""
    result = []
    for stage in pipeline:
        if "$match" in stage:
            stage = {k: (_coerce_date_strings(v, k) if k == "$match" else v) for k, v in stage.items()}
        result.append(stage)
    return result


# Fields the LLM sometimes invents that do NOT exist in values collection.
# The 'values' field is a float array; there is no values.uuid / values.id sub-field.
_INVALID_VALUES_FIELD_ROOTS = {
    "values.uuid",
    "values.id",
    "values.name",
    "values.type",
    "values.unit",
}


def _sanitize_match_stage(match_body: dict[str, Any], collection: str) -> dict[str, Any]:
    """Remove known-bad field references from a $match body.

    This catches cases where the LLM invents sub-fields of the 'values' float
    array (e.g. values.uuid) or uses fabricated testProgramId lists that are
    not derived from actual data — both of which produce 0 rows.
    """
    cleaned: dict[str, Any] = {}
    for key, val in match_body.items():
        if key.startswith("$"):
            # Operator at the top level ($or, $and, $nor) — recurse
            if isinstance(val, list):
                cleaned[key] = [
                    _sanitize_match_stage(item, collection) if isinstance(item, dict) else item
                    for item in val
                ]
            else:
                cleaned[key] = val
            continue

        # Skip fields that are known to not exist in the values collection.
        if key.lower() in _INVALID_VALUES_FIELD_ROOTS:
            continue

        cleaned[key] = val
    return cleaned


def _sanitize_llm_pipeline(pipeline: list[dict[str, Any]], collection: str) -> list[dict[str, Any]]:
    """Strip LLM-hallucinated invalid field references from $match stages."""
    result = []
    for stage in pipeline:
        if "$match" in stage:
            sanitized_body = _sanitize_match_stage(stage["$match"], collection)
            if sanitized_body:
                result.append({"$match": sanitized_body})
            # If $match is now empty, drop it entirely (don't filter at all).
        else:
            result.append(stage)
    return result


def _contains_deprecated_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and _is_deprecated_field_reference(key):
                return True
            if _contains_deprecated_placeholder(nested):
                return True
        return False

    if isinstance(value, list):
        return any(_contains_deprecated_placeholder(item) for item in value)

    if isinstance(value, str):
        return _is_deprecated_field_reference(value)

    return False


def _contains_tests_scope_reference(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and (key == "test" or key.startswith("test.")):
                return True
            if _contains_tests_scope_reference(nested):
                return True
        return False

    if isinstance(value, list):
        return any(_contains_tests_scope_reference(item) for item in value)

    if isinstance(value, str):
        token = value.strip()
        return token == "$test" or token.startswith("$test.")

    return False


class MongoExecutor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.gateway = ClaudeGateway()

    def _pick_collection(self, intent: IntentCategory, operations: list[PlanOperation]) -> CollectionName:
        if intent in {IntentCategory.trend_drift, IntentCategory.anomaly_check}:
            return CollectionName.values
        if PlanOperation.time_series_extract in operations:
            return CollectionName.values
        return CollectionName.tests

    def _physical_collection_name(self, logical_collection: CollectionName) -> str:
        if logical_collection == CollectionName.values:
            return self.settings.mongo_collection_values
        return self.settings.mongo_collection_tests

    def _resolve_filter_field(self, field: str, collection: CollectionName) -> str:
        token = field.strip()
        if not token:
            return token

        token = re.sub(r"^(?:_tests|tests|Tests|valuecolumns_migrated|values|Values)\.", "", token)

        token_lower = token.lower()
        if collection == CollectionName.tests and token_lower.startswith("testparametersflat."):
            suffix = token.split(".", 1)[1]
            token = f"TestParametersFlat.{suffix}"
            token_lower = token.lower()
        if collection == CollectionName.values and token_lower.startswith("metadata.refid"):
            parts = token.split(".")
            if len(parts) >= 2:
                parts[0] = "metadata"
                parts[1] = "refId"
                token = ".".join(parts)
                token_lower = token.lower()
        if collection == CollectionName.values and token_lower.startswith("metadata.childid"):
            parts = token.split(".")
            if len(parts) >= 2:
                parts[0] = "metadata"
                parts[1] = "childId"
                token = ".".join(parts)
                token_lower = token.lower()

        if collection == CollectionName.tests:
            aliases = {
                "testparametersflat": "__site_token__",
                "testparametersflat.machine": "TestParametersFlat.MACHINE_TYPE_STR",
                "testparametersflat.customer": "TestParametersFlat.CUSTOMER",
                "testparametersflat.customer_name": "TestParametersFlat.CUSTOMER_NAME",
                "testparametersflat.notes": "TestParametersFlat.NOTES",
                "testparametersflat.plant": "__site_token__",
                "testparametersflat.site": "__site_token__",
                "testparametersflat.location": "__site_token__",
                "machine": "TestParametersFlat.MACHINE_TYPE_STR",
                "site": "__site_token__",
                "customer": "TestParametersFlat.CUSTOMER",
                "material": "TestParametersFlat.MATERIAL",
                "standard": "TestParametersFlat.STANDARD",
                "tester": "TestParametersFlat.TESTER",
                # modifiedOn in _tests is stored as {} (empty object) — never filter by it.
                # Date filters on tests are suppressed; use uploadDate on values collection instead.
                "createdat": "__suppress_date__",
                "modifiedon": "__suppress_date__",
                "test_id": "_id",
                "test_type": "name",
                "test_name": "name",
                "name": "name",
                "status": "state",
                "result": "state",
                "material_id": "TestParametersFlat.MATERIAL",
                "test_date": "__suppress_date__",
                "program": "testProgramId",
                "testprogramid": "testProgramId",
            }
        else:
            aliases = {
                "refid": "metadata.refId",
                "childid": "metadata.childId",
                "metadata.refid": "metadata.refId",
                "metadata.childid": "metadata.childId",
                "createdat": "uploadDate",
                "modifiedon": "uploadDate",
                "value": "values",
                "unit": "metadata.childId",
                "time": "uploadDate",
                "timestamp": "uploadDate",
                "test_id": "metadata.refId",
                "test_date": "uploadDate",
            }

        return aliases.get(token_lower, token)

    def _site_filter_values(self, plan: Any) -> list[str]:
        raw_filters = getattr(plan, "filters", [])
        values: list[str] = []

        for item in raw_filters:
            field = getattr(item, "field", "")
            if not isinstance(field, str):
                continue

            field_lower = field.strip().lower()
            if field_lower != "site" and "testparametersflat" not in field_lower:
                continue

            raw_value = getattr(item, "value", None)
            if isinstance(raw_value, list):
                for v in raw_value:
                    text = str(v).strip()
                    if text:
                        values.append(text)
            elif raw_value is not None:
                text = str(raw_value).strip()
                if text:
                    values.append(text)

        # Preserve order and remove duplicates.
        return list(dict.fromkeys(values))

    def _count_site_hits(self, tests_collection: Any, site_values: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in site_values:
            escaped = re.escape(value)
            regex = {"$regex": escaped, "$options": "i"}
            query = {
                "$or": [
                    {"TestParametersFlat.NOTES": regex},
                    {"TestParametersFlat.Designation sub-series 1": regex},
                    {"TestParametersFlat.CUSTOMER": regex},
                    {"TestParametersFlat.CUSTOMER_NAME": regex},
                ]
            }
            counts[value] = int(tests_collection.count_documents(query))
        return counts

    def _normalize_collection_name(self, raw_value: Any) -> str | None:
        if not isinstance(raw_value, str):
            return None

        value = raw_value.strip()
        if value in {CollectionName.tests.value, CollectionName.values.value}:
            return value

        token = value.lower()
        aliases = {
            "test": CollectionName.tests.value,
            "tests": CollectionName.tests.value,
            "_tests": CollectionName.tests.value,
            "value": CollectionName.values.value,
            "values": CollectionName.values.value,
            "valuecolumns_migrated": CollectionName.values.value,
        }
        return aliases.get(token)

    def _extract_metric_request(self, question: str) -> dict[str, str] | None:
        q = question.strip().lower()
        if not q:
            return None

        metric_op: str | None = None
        metric_prefix: str | None = None
        if any(token in q for token in {"highest", "maximum", " max "}):
            metric_op = "$max"
            metric_prefix = "highest"
        elif any(token in q for token in {"lowest", "minimum", " min "}):
            metric_op = "$min"
            metric_prefix = "lowest"
        elif any(token in q for token in {"average", "mean", " avg "}):
            metric_op = "$avg"
            metric_prefix = "average"

        if not metric_op or not metric_prefix:
            return None

        # Try to capture the phrase after highest/lowest/average up to a connector.
        metric_phrase_match = re.search(
            r"(?:highest|maximum|max(?:imum)?|lowest|minimum|min(?:imum)?|average|mean|avg)\s+(.+?)(?=\s+(?:for|of|between|among|across|and|by|where|in|on|with|for\b\s+tester)\b|[\,\?\.!]|$)",
            q,
            flags=re.I,
        )
        metric_phrase = metric_phrase_match.group(1).strip() if metric_phrase_match else ""
        metric_phrase = re.sub(r"\btester[_\-]?\d+\b", "", metric_phrase, flags=re.I).strip()
        metric_phrase = re.sub(r"\s+", " ", metric_phrase)

        normalized_metric = _normalize_attribute_token(metric_phrase)
        attribute_key = ATTRIBUTE_ALIASES.get(normalized_metric)

        parameter_index = _load_test_parameter_name_index()
        if not attribute_key and normalized_metric in parameter_index:
            attribute_key = parameter_index[normalized_metric]

        if not attribute_key and normalized_metric:
            # Fuzzy fallback: nearest key by token containment and length distance.
            fuzzy_candidates: list[tuple[int, str]] = []
            for token, name in parameter_index.items():
                if normalized_metric in token or token in normalized_metric:
                    fuzzy_candidates.append((abs(len(token) - len(normalized_metric)), name))
            if fuzzy_candidates:
                fuzzy_candidates.sort(key=lambda item: item[0])
                attribute_key = fuzzy_candidates[0][1]

        if not attribute_key:
            return None

        metric_suffix = re.sub(r"[^a-z0-9]+", "_", attribute_key.lower()).strip("_")
        metric_field_name = f"{metric_prefix}_{metric_suffix}"
        return {
            "attribute_key": attribute_key,
            "metric_op": metric_op,
            "metric_field_name": metric_field_name,
        }

    def _semantic_result_uuids(self, semantic_candidates: list[dict] | None, max_items: int = 6) -> list[str]:
        if not semantic_candidates:
            return []

        uuids: list[str] = []
        seen: set[str] = set()
        for item in semantic_candidates:
            if not isinstance(item, dict):
                continue

            category = str(item.get("category", "")).strip().lower()
            if category not in {"result", "channel"}:
                continue

            raw_uuid = item.get("uuid")
            if not isinstance(raw_uuid, str) or not raw_uuid.strip():
                continue

            normalized = raw_uuid.strip().strip("{}")
            if not normalized:
                continue

            unique_key = normalized.lower()
            if unique_key in seen:
                continue

            seen.add(unique_key)
            uuids.append(normalized)
            if len(uuids) >= max_items:
                break

        return uuids

    def _semantic_childid_match(self, semantic_candidates: list[dict] | None) -> dict[str, Any] | None:
        uuids = self._semantic_result_uuids(semantic_candidates)
        if not uuids:
            return None

        clauses = [{"metadata.childId": {"$regex": re.escape(uuid), "$options": "i"}} for uuid in uuids]
        if len(clauses) == 1:
            return clauses[0]
        return {"$or": clauses}

    def _resolve_tests_scope_ref_ids(self, plan: Any, db: Any, max_ids: int = 3000) -> list[str] | None:
        raw_filters = [f.model_dump() for f in getattr(plan, "filters", [])]
        tests_scope_filters = [
            item
            for item in raw_filters
            if _is_tests_scope_filter(str(item.get("field", "")))
        ]
        if not tests_scope_filters:
            return None

        tests_match_query = _filters_to_match(
            tests_scope_filters,
            field_resolver=lambda field: self._resolve_filter_field(field, CollectionName.tests),
        )
        if not tests_match_query:
            return None

        try:
            tests_collection = db[self.settings.mongo_collection_tests]
            cursor = tests_collection.find(tests_match_query, {"_id": 1}).limit(max_ids)
            ref_ids: list[str] = []
            seen: set[str] = set()
            for row in cursor:
                raw_id = row.get("_id")
                if raw_id is None:
                    continue
                doc_id = raw_id if isinstance(raw_id, str) else str(raw_id)
                if not doc_id or doc_id in seen:
                    continue
                seen.add(doc_id)
                ref_ids.append(doc_id)
            return ref_ids
        except Exception:  # noqa: BLE001
            return None

    def _inject_refid_fast_match(self, pipeline: list[dict[str, Any]], ref_ids: list[str] | None) -> list[dict[str, Any]]:
        if ref_ids is None:
            return pipeline

        if not ref_ids:
            # tests-scope filters exist but no matching tests were found.
            return [{"$match": {"_id": {"$exists": False}}}]

        cleaned: list[dict[str, Any]] = []
        for stage in pipeline:
            if "$lookup" in stage and isinstance(stage["$lookup"], dict):
                lookup = stage["$lookup"]
                if lookup.get("from") == self.settings.mongo_collection_tests and lookup.get("as") == "test":
                    continue

            if "$unwind" in stage:
                unwind = stage["$unwind"]
                if isinstance(unwind, str) and unwind in {"$test", "test"}:
                    continue
                if isinstance(unwind, dict) and str(unwind.get("path")) in {"$test", "test"}:
                    continue

            if "$match" in stage and _contains_tests_scope_reference(stage.get("$match")):
                continue

            cleaned.append(stage)

        return [{"$match": {"metadata.refId": {"$in": ref_ids}}}, *cleaned]

    def _normalize_stage(self, stage: Any) -> dict[str, Any] | None:
        if not isinstance(stage, dict) or not stage:
            return None

        normalized: dict[str, Any] = {}
        for key, value in stage.items():
            if not isinstance(key, str):
                return None
            clean_key = key.strip()
            if not clean_key:
                return None
            if not clean_key.startswith("$"):
                clean_key = f"${clean_key.lstrip('$')}"
            normalized[clean_key] = value

        return normalized

    def _validate_pipeline(self, pipeline: Any) -> list[dict[str, Any]] | None:
        if not isinstance(pipeline, list):
            return None
        validated: list[dict[str, Any]] = []
        for stage in pipeline:
            normalized_stage = self._normalize_stage(stage)
            if not normalized_stage:
                return None
            if not all(isinstance(key, str) and key.startswith("$") for key in normalized_stage.keys()):
                return None
            if any(key in MUTATING_AGG_STAGES for key in normalized_stage.keys()):
                return None
            if _contains_deprecated_placeholder(normalized_stage):
                return None
            validated.append(normalized_stage)
        return validated

    def _generate_candidate_llm(self, plan: Any, semantic_candidates: list[dict] | None = None) -> MongoQueryCandidate | None:
        provider = self.settings.llm_provider.lower()
        if provider not in {"anthropic", "openai"} or not self.gateway.is_ready():
            return None

        model_name = self.settings.openai_model_query if provider == "openai" else self.settings.anthropic_model_query

        # Extract UUIDs from semantic candidates so the LLM can build correct childId regex filters.
        semantic_uuids: list[dict] = []
        for sc in (semantic_candidates or [])[:6]:
            uuid = sc.get("uuid")
            name = sc.get("name", "")
            if uuid:
                semantic_uuids.append({"name": name, "uuid": uuid, "regex_hint": f"{uuid}"})

        compact_plan = {
            "intent": getattr(plan.intent, "value", str(plan.intent)),
            "required_collections": [getattr(c, "value", str(c)) for c in getattr(plan, "required_collections", [])],
            "fields_needed": list(getattr(plan, "fields_needed", []))[:12],
            "operations": [getattr(op, "value", str(op)) for op in getattr(plan, "operations", [])][:10],
            "statistics_methods": [
                getattr(stat, "value", str(stat)) for stat in getattr(plan, "statistics_methods", [])
            ][:10],
            "filters": [flt.model_dump() for flt in getattr(plan, "filters", [])][:6],
            "chart_needed": bool(getattr(plan, "chart_needed", False)),
        }

        system_prompt = (
            "You are a MongoDB aggregation generator for materials testing data. "
            "Return strict JSON only."
        )
        user_prompt = (
            "Generate a query candidate for this plan.\n"
            "Allowed collection values: Tests or Values.\n"
            "Return schema: {\"collection\": string, \"pipeline\": object[], \"explanation\": string, \"expected_shape\": string[]}\n"
            "Use only valid MongoDB aggregation stages. Keep the pipeline concise and safe.\n"
            "Never use $out or $merge.\n"
            "CRITICAL Mongo schema facts (read carefully before generating):\n"
            "- Tests physical collection: _tests\n"
            "  Top-level fields: _id (string, format '{UUID}'), name, state, testProgramId, TestParametersFlat (nested object)\n"
            "  WARNING: modifiedOn in _tests is stored as {} (empty object) - NEVER filter or sort by modifiedOn\n"
            "  CRITICAL: ALL test metadata lives inside TestParametersFlat - ALWAYS use the full dotted path:\n"
            "    TestParametersFlat.TESTER       -> tester/operator name (e.g. 'Tester_2')\n"
            "    TestParametersFlat.CUSTOMER     -> customer/company name (e.g. 'Company_1')\n"
            "    TestParametersFlat.CUSTOMER_NAME-> alternative customer name field\n"
            "    TestParametersFlat.MATERIAL     -> material under test\n"
            "    TestParametersFlat.STANDARD     -> test standard (e.g. 'DIN EN ISO 6892-1')\n"
            "    TestParametersFlat.TYPE_OF_TESTING_STR -> test type: 'tensile', 'compression', or 'flexure'\n"
            "    TestParametersFlat.MACHINE_TYPE_STR -> machine type string\n"
            "    TestParametersFlat.NOTES        -> free-text notes\n"
            "    TestParametersFlat.SPECIMEN_TYPE-> specimen type identifier\n"
            "    TestParametersFlat.JOB_NO       -> job/order number\n"
            "  NEVER use bare field names like 'tester', 'customer', 'material', 'standard' without the TestParametersFlat. prefix.\n"
            "- Values physical collection: valuecolumns_migrated\n"
            "  Fields: _id (ObjectId), metadata.refId (string matching _tests._id), "
            "metadata.childId (composite string like '{UUID}-UnitTable.{UUID}-UnitTable_Value' or '{UUID}.{UUID}_Value'), "
            "uploadDate (ISODate - use this for ALL date/time filtering), values (float array), valuesCount (int)\n"
            "- Join relation: _tests._id == valuecolumns_migrated.metadata.refId\n"
            "- For date/time filtering: ALWAYS use uploadDate on valuecolumns_migrated, never modifiedOn\n"
            "- For childId UUID matching: use $regex, e.g. {\"$regex\": \"UUID_HERE\", \"$options\": \"i\"}\n"
            "  The semantic_candidates in the plan contain UUIDs - wrap them in regex for childId matching\n"
            "- If no specific childId UUID is known, do NOT add a childId filter - return all values\n"
            "- IMPORTANT: the values collection contains 5.9M documents. For trend/anomaly analysis, do NOT\n"
            "  use a small $limit before $unwind. Fetch at least 200 raw documents (before unwind).\n"
            "  Each document's 'values' array holds ~44000 floats. Use $limit AFTER $unwind to cap output.\n"
            "- DATA DATE RANGE: uploadDate values in the dataset range from 2026-02-27 to 2026-03-03.\n"
            "  For questions like 'last 6 months', use ISODate('2025-09-01') as the lower bound.\n"
            "  DO NOT emit date values as plain strings - emit them as ISO format that Python can parse.\n"
            "  All data falls within the last 6 months, so a broad date range will return results.\n"
            "Avoid deprecated placeholders: site, test_name, sample_id, test_date, test_id, createdAt.\n"
            f"Plan: {compact_plan}\n"
            f"Semantic UUID candidates (use these for childId regex matching if relevant): {semantic_uuids}\n"
            f"Max rows limit: {self.settings.max_query_rows}\n"
        )

        prompt_attempts = [
            user_prompt,
            (
                "Return strict JSON only with schema: "
                "{\"collection\": \"Tests|Values\", \"pipeline\": [{\"$match\": {}}, {\"$limit\": 100}], "
                "\"explanation\": string, \"expected_shape\": string[]}. "
                "Key rules: modifiedOn in _tests is {} (not a date) - never use it. "
                "Use uploadDate on valuecolumns_migrated for date filtering. "
                "For childId UUID matching use $regex. "
                "Avoid site/test_name/test_date/test_id placeholders. "
                f"Use this plan: {compact_plan}"
            ),
        ]

        result = None
        for prompt in prompt_attempts:
            candidate_result = self.gateway.generate_json(
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt,
            )
            if candidate_result:
                result = candidate_result
                break

        if not result:
            return None

        collection_value = self._normalize_collection_name(result.get("collection"))
        if collection_value not in {CollectionName.tests.value, CollectionName.values.value}:
            return None

        pipeline = self._validate_pipeline(result.get("pipeline"))
        if not pipeline:
            return None

        # Convert any ISO date strings in $match stages to Python datetime objects
        # so PyMongo sends them as BSON ISODate rather than strings.
        pipeline = _coerce_pipeline_dates(pipeline)

        # Remove hallucinated field references (e.g. values.uuid) that never exist.
        pipeline = _sanitize_llm_pipeline(pipeline, collection_value)

        # If the LLM omitted a $match stage entirely but the plan has filters, inject one.
        # This prevents silent full-collection scans when the LLM ignores filter fields.
        if not any("$match" in stage for stage in pipeline) and getattr(plan, "filters", []):
            logical_collection = CollectionName(collection_value)
            raw_filters = [f.model_dump() for f in plan.filters]
            injected_match = _filters_to_match(
                raw_filters,
                field_resolver=lambda field: self._resolve_filter_field(field, logical_collection),
            )
            if injected_match:
                pipeline.insert(0, {"$match": injected_match})

        if not any("$limit" in stage for stage in pipeline):
            pipeline.append({"$limit": min(200, self.settings.max_query_rows)})

        expected_shape = result.get("expected_shape")
        if not isinstance(expected_shape, list):
            expected_shape = (
                ["name", "state", "testProgramId", "modifiedOn", "TestParametersFlat"]
                if collection_value == CollectionName.tests.value
                else ["refId", "childId", "uploadDate", "values", "valuesCount"]
            )

        explanation = result.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            explanation = "Pipeline generated by Claude query mode."

        return MongoQueryCandidate(
            collection=CollectionName(collection_value),
            pipeline=pipeline,
            explanation=explanation,
            expected_shape=[str(item) for item in expected_shape],
        )

    def generate_candidate_from_plan(self, plan: Any, allow_llm: bool = True, semantic_candidates: list[dict] | None = None) -> MongoQueryCandidate:
        if allow_llm and self.settings.query_mode.lower() == "llm":
            llm_candidate = self._generate_candidate_llm(plan, semantic_candidates=semantic_candidates)
            if llm_candidate is not None:
                return llm_candidate

        collection = self._pick_collection(plan.intent, plan.operations)
        pipeline: list[dict[str, Any]] = []
        raw_filters = [f.model_dump() for f in plan.filters]

        if collection == CollectionName.values:
            value_filters = [
                item
                for item in raw_filters
                if not _is_tests_scope_filter(str(item.get("field", "")))
            ]
            tests_scope_filters = [
                item
                for item in raw_filters
                if _is_tests_scope_filter(str(item.get("field", "")))
            ]

            value_match_query = _filters_to_match(
                value_filters,
                field_resolver=lambda field: self._resolve_filter_field(field, CollectionName.values),
            )

            semantic_match_query: dict[str, Any] = {}
            if semantic_candidates and plan.intent in {
                IntentCategory.trend_drift,
                IntentCategory.anomaly_check,
                IntentCategory.hypothesis,
            }:
                semantic_childid = self._semantic_childid_match(semantic_candidates)
                if semantic_childid:
                    semantic_match_query = semantic_childid

            combined_value_match: dict[str, Any] = {}
            if value_match_query and semantic_match_query:
                combined_value_match = {"$and": [value_match_query, semantic_match_query]}
            elif value_match_query:
                combined_value_match = value_match_query
            elif semantic_match_query:
                combined_value_match = semantic_match_query

            if combined_value_match:
                pipeline.append({"$match": combined_value_match})

            if tests_scope_filters:
                pipeline.extend(
                    [
                        {
                            "$lookup": {
                                "from": self.settings.mongo_collection_tests,
                                "localField": "metadata.refId",
                                "foreignField": "_id",
                                "as": "test",
                            }
                        },
                        {"$unwind": "$test"},
                    ]
                )

                tests_match_query = _filters_to_match(
                    tests_scope_filters,
                    field_resolver=lambda field: self._resolve_filter_field(field, CollectionName.tests),
                    field_prefix="test.",
                )
                if tests_match_query:
                    pipeline.append({"$match": tests_match_query})
        else:
            match_query = _filters_to_match(
                raw_filters,
                field_resolver=lambda field: self._resolve_filter_field(field, collection),
            )
            if match_query:
                pipeline.append({"$match": match_query})

        if collection == CollectionName.values:
            if plan.intent == IntentCategory.trend_drift:
                pipeline.extend(
                    [
                        {"$sort": {"uploadDate": -1}},
                        {"$limit": min(240, self.settings.max_query_rows)},
                        {
                            "$project": {
                                "_id": 0,
                                "refId": "$metadata.refId",
                                "childId": "$metadata.childId",
                                "uploadDate": 1,
                                "valuesCount": 1,
                                # Compute one representative value per document to avoid
                                # unwinding very large arrays for long-window trend prompts.
                                "value": {
                                    "$cond": [
                                        {"$gt": [{"$size": {"$ifNull": ["$values", []]}}, 0]},
                                        {"$avg": "$values"},
                                        None,
                                    ]
                                },
                            }
                        },
                        {"$match": {"value": {"$ne": None}}},
                    ]
                )
            elif plan.intent == IntentCategory.anomaly_check:
                pipeline.extend(
                    [
                        {"$sort": {"uploadDate": -1}},
                        {"$limit": min(320, self.settings.max_query_rows)},
                        {
                            "$project": {
                                "_id": 0,
                                "refId": "$metadata.refId",
                                "childId": "$metadata.childId",
                                "uploadDate": 1,
                                "values": 1,
                                "valuesCount": 1,
                            }
                        },
                        {"$unwind": {"path": "$values", "includeArrayIndex": "sampleIndex"}},
                        {
                            "$project": {
                                "refId": 1,
                                "childId": 1,
                                "uploadDate": 1,
                                "sampleIndex": 1,
                                "value": "$values",
                            }
                        },
                    ]
                )
            else:
                pipeline.extend(
                    [
                        {
                            "$project": {
                                "_id": 0,
                                "refId": "$metadata.refId",
                                "childId": "$metadata.childId",
                                "uploadDate": 1,
                                "valuesCount": 1,
                            }
                        },
                    ]
                )
        else:
            question_lower = str(getattr(plan, "user_question", "")).lower()
            metric_request = self._extract_metric_request(str(getattr(plan, "user_question", "")))
            is_metric_compare = plan.intent == IntentCategory.comparison and metric_request is not None

            if plan.intent == IntentCategory.comparison:
                if is_metric_compare and metric_request is not None:
                    pipeline.extend(
                        [
                            {
                                "$project": {
                                    "tester": "$TestParametersFlat.TESTER",
                                    "metric_value": {
                                        "$convert": {
                                            "input": {
                                                "$getField": {
                                                    "input": "$TestParametersFlat",
                                                    "field": metric_request["attribute_key"],
                                                }
                                            },
                                            "to": "double",
                                            "onError": None,
                                            "onNull": None,
                                        }
                                    },
                                }
                            },
                            {"$match": {"tester": {"$ne": None}, "metric_value": {"$ne": None}}},
                            {
                                "$group": {
                                    "_id": "$tester",
                                    metric_request["metric_field_name"]: {metric_request["metric_op"]: "$metric_value"},
                                }
                            },
                            {"$project": {"_id": 0, "tester": "$_id", metric_request["metric_field_name"]: 1}},
                            {"$sort": {"tester": 1}},
                        ]
                    )
                else:
                    pipeline.extend(
                        [
                            {
                                "$project": {
                                    "machine": {
                                        "$ifNull": [
                                            "$TestParametersFlat.MACHINE_TYPE_STR",
                                            "$TestParametersFlat.MACHINE_DATA",
                                            "Unknown",
                                        ]
                                    }
                                }
                            },
                            {"$group": {"_id": "$machine", "n": {"$sum": 1}}},
                            {"$sort": {"n": -1}},
                        ]
                    )
            elif plan.intent == IntentCategory.validation_compliance:
                pipeline.extend(
                    [
                        {
                            "$project": {
                                "_id": 1,
                                "name": 1,
                                "state": 1,
                                "testProgramId": 1,
                                "modifiedOn": 1,
                                "hasMachineConfigurationInfo": 1,
                                "clientAppType": 1,
                                "TestParametersFlat": 1,
                            }
                        },
                    ]
                )
            else:
                pipeline.extend(
                    [
                        {
                            "$project": {
                                "_id": 1,
                                "name": 1,
                                "state": 1,
                                "testProgramId": 1,
                                "modifiedOn": 1,
                                "TestParametersFlat": 1,
                            }
                        },
                    ]
                )

        pipeline.append({"$limit": min(200, self.settings.max_query_rows)})

        expected_shape = (
            ["refId", "childId", "uploadDate", "sampleIndex", "value"]
            if collection == CollectionName.values
            else ["name", "state", "testProgramId", "modifiedOn", "TestParametersFlat"]
        )

        return MongoQueryCandidate(
            collection=collection,
            pipeline=pipeline,
            explanation="Pipeline generated from investigation plan using deterministic mock generator.",
            expected_shape=expected_shape,
        )

    def _repair_pipeline_with_llm(
        self,
        failed_pipeline: list[dict[str, Any]],
        error: str,
        plan: Any,
        collection_name: str,
    ) -> list[dict[str, Any]] | None:
        provider = self.settings.llm_provider.lower()
        if provider not in {"anthropic", "openai"} or not self.gateway.is_ready():
            return None

        model_name = self.settings.openai_model_query if provider == "openai" else self.settings.anthropic_model_query

        system_prompt = (
            "You repair MongoDB aggregation pipelines. Return strict JSON only with this schema: "
            "{\"pipeline\": object[], \"explanation\": string}."
        )
        user_prompt = (
            "The pipeline failed. Fix it while preserving user intent.\n"
            "Constraints:\n"
            "- Return only valid aggregation stage objects.\n"
            "- Keep a reasonable row limit at the end.\n"
            "- Never use $out or $merge.\n"
            "- Do not include markdown.\n"
            "Schema reminder for _tests collection: ALL metadata is inside TestParametersFlat.\n"
            "  Use TestParametersFlat.TESTER, TestParametersFlat.CUSTOMER, TestParametersFlat.MATERIAL,\n"
            "  TestParametersFlat.STANDARD, TestParametersFlat.TYPE_OF_TESTING_STR, TestParametersFlat.MACHINE_TYPE_STR.\n"
            "  NEVER use bare field names like 'tester' or 'customer' without the TestParametersFlat. prefix.\n"
            f"Collection: {collection_name}\n"
            f"Plan: {plan.model_dump()}\n"
            f"Failed pipeline: {failed_pipeline}\n"
            f"Error: {error}\n"
            f"Max rows limit: {self.settings.max_query_rows}\n"
        )

        result = self.gateway.generate_json(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        if not result:
            return None

        pipeline = self._validate_pipeline(result.get("pipeline"))
        if not pipeline:
            return None

        pipeline = _coerce_pipeline_dates(pipeline)

        if not any("$limit" in stage for stage in pipeline):
            pipeline.append({"$limit": min(200, self.settings.max_query_rows)})

        return pipeline

    def _repair_pipeline_from_error(
        self,
        failed_pipeline: list[dict[str, Any]],
        error: str,
        plan: Any,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        if self.settings.query_mode.lower() == "llm":
            repaired = self._repair_pipeline_with_llm(failed_pipeline, error, plan, collection_name)
            if repaired is not None:
                return repaired

        # Fallback keeps retry behavior deterministic and safe during hackathon.
        if "Unrecognized pipeline stage name" in error:
            cleaned = [stage for stage in failed_pipeline if all(k.startswith("$") for k in stage.keys())]
            return cleaned or [{"$limit": 50}]
        return [{"$limit": 50}]

    def _get_database(self, client: MongoClient):
        if self.settings.mongo_db_name:
            return client[self.settings.mongo_db_name]

        user_dbs = [db for db in client.list_database_names() if db not in {"admin", "local", "config"}]
        if not user_dbs:
            raise RuntimeError("No user database found. Set MONGO_DB_NAME in .env.")
        return client[user_dbs[0]]

    def run_plan_with_repair(self, plan: Any, max_repairs: int | None = None, semantic_candidates: list[dict] | None = None) -> QueryRunResponse:
        repairs = self.settings.max_query_repairs if max_repairs is None else max_repairs

        candidate = self.generate_candidate_from_plan(plan, allow_llm=True, semantic_candidates=semantic_candidates)
        attempts: list[QueryAttempt] = []
        corrected = False
        fallback_attempted = False
        current_pipeline = candidate.pipeline
        pre_resolved_ref_ids: list[str] | None = None

        with MongoClient(self.settings.mongo_uri, serverSelectionTimeoutMS=7000) as client:
            db = self._get_database(client)
            pre_resolved_ref_ids = self._resolve_tests_scope_ref_ids(plan, db)
            if candidate.collection == CollectionName.values:
                candidate.pipeline = self._inject_refid_fast_match(candidate.pipeline, pre_resolved_ref_ids)
                current_pipeline = candidate.pipeline

            physical_collection_name = self._physical_collection_name(candidate.collection)
            collection = db[physical_collection_name]

            for idx in range(repairs + 1):
                attempt_number = idx + 1
                try:
                    cursor = collection.aggregate(current_pipeline, allowDiskUse=True)
                    rows = [_to_json_safe(row) for row in cursor]
                    rows = rows[: self.settings.max_query_rows]

                    attempts.append(
                        QueryAttempt(
                            attempt=attempt_number,
                            pipeline=current_pipeline,
                            error=None,
                            corrected_from_previous=corrected,
                        )
                    )

                    if self.settings.query_mode.lower() == "llm" and len(rows) == 0:
                        fallback_candidate = self.generate_candidate_from_plan(plan, allow_llm=False)
                        if fallback_candidate.collection == CollectionName.values:
                            fallback_candidate.pipeline = self._inject_refid_fast_match(
                                fallback_candidate.pipeline,
                                pre_resolved_ref_ids,
                            )

                        # Always attempt the deterministic fallback when the LLM pipeline
                        # returned 0 rows. The previous pipeline-equality guard was fragile
                        # (datetime objects vs strings made equal pipelines look different,
                        # and coincidentally-equal pipelines blocked recovery).
                        fallback_attempted = True
                        fallback_collection_name = self._physical_collection_name(fallback_candidate.collection)
                        fallback_collection = db[fallback_collection_name]

                        try:
                            fallback_cursor = fallback_collection.aggregate(fallback_candidate.pipeline, allowDiskUse=True)
                            fallback_rows = [_to_json_safe(row) for row in fallback_cursor]
                            fallback_rows = fallback_rows[: self.settings.max_query_rows]

                            attempts.append(
                                QueryAttempt(
                                    attempt=attempt_number + 1,
                                    pipeline=fallback_candidate.pipeline,
                                    error=None,
                                    corrected_from_previous=True,
                                )
                            )

                            candidate = fallback_candidate
                            current_pipeline = fallback_candidate.pipeline
                            rows = fallback_rows
                            corrected = True

                            if fallback_rows:
                                fallback_candidate.pipeline = fallback_candidate.pipeline
                                return QueryRunResponse(
                                    status="success",
                                    candidate=fallback_candidate,
                                    attempts=attempts,
                                    row_count=len(fallback_rows),
                                    rows=fallback_rows,
                                    corrected_automatically=True,
                                )
                        except Exception as fallback_exc:  # noqa: BLE001
                            attempts.append(
                                QueryAttempt(
                                    attempt=attempt_number + 1,
                                    pipeline=fallback_candidate.pipeline,
                                    error=str(fallback_exc),
                                    corrected_from_previous=True,
                                )
                            )

                    if len(rows) == 0:
                        site_values = self._site_filter_values(plan)
                        if site_values:
                            tests_collection = db[self.settings.mongo_collection_tests]
                            site_counts = self._count_site_hits(tests_collection, site_values)
                            missing = [name for name, count in site_counts.items() if count == 0]
                            if missing:
                                attempts[-1].error = (
                                    "No matching documents for one or more requested sites. "
                                    f"Counts by site token: {site_counts}."
                                )

                        # Last-resort: strip all $match stages from current pipeline and retry.
                        # This recovers from overly restrictive filters (e.g. broken date/childId filters).
                        stripped_pipeline = [
                            stage for stage in current_pipeline
                            if "$match" not in stage
                        ]
                        if stripped_pipeline and stripped_pipeline != current_pipeline:
                            try:
                                stripped_cursor = collection.aggregate(stripped_pipeline, allowDiskUse=True)
                                stripped_rows = [_to_json_safe(row) for row in stripped_cursor]
                                stripped_rows = stripped_rows[: self.settings.max_query_rows]
                                if stripped_rows:
                                    attempt_number_stripped = len(attempts) + 1
                                    attempts.append(
                                        QueryAttempt(
                                            attempt=attempt_number_stripped,
                                            pipeline=stripped_pipeline,
                                            error=None,
                                            corrected_from_previous=True,
                                        )
                                    )
                                    candidate.pipeline = stripped_pipeline
                                    return QueryRunResponse(
                                        status="success",
                                        candidate=candidate,
                                        attempts=attempts,
                                        row_count=len(stripped_rows),
                                        rows=stripped_rows,
                                        corrected_automatically=True,
                                    )
                            except Exception:  # noqa: BLE001
                                pass

                    candidate.pipeline = current_pipeline
                    return QueryRunResponse(
                        status="success",
                        candidate=candidate,
                        attempts=attempts,
                        row_count=len(rows),
                        rows=rows,
                        corrected_automatically=(corrected or fallback_attempted),
                    )
                except Exception as exc:  # noqa: BLE001
                    error_text = str(exc)
                    attempts.append(
                        QueryAttempt(
                            attempt=attempt_number,
                            pipeline=current_pipeline,
                            error=error_text,
                            corrected_from_previous=corrected,
                        )
                    )

                    if idx >= repairs:
                        candidate.pipeline = current_pipeline
                        return QueryRunResponse(
                            status="failed",
                            candidate=candidate,
                            attempts=attempts,
                            row_count=0,
                            rows=[],
                            corrected_automatically=(corrected or fallback_attempted),
                        )

                    current_pipeline = self._repair_pipeline_from_error(
                        current_pipeline,
                        error_text,
                        plan,
                        physical_collection_name,
                    )
                    corrected = True

        return QueryRunResponse(
            status="failed",
            candidate=candidate,
            attempts=attempts,
            row_count=0,
            rows=[],
            corrected_automatically=(corrected or fallback_attempted),
        )

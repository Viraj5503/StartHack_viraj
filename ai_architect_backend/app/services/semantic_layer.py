import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


def _normalize(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


@dataclass
class SemanticDictionary:
    channels_by_name: dict[str, dict[str, Any]]
    results_by_name: dict[str, list[dict[str, Any]]]
    params_by_name: dict[str, dict[str, Any]]


def _default_helper_root() -> Path:
    local_root = Path(__file__).resolve().parents[2] / "resources" / "uuid_helpers"
    if local_root.exists():
        return local_root

    legacy_root = Path(__file__).resolve().parents[3] / "ZwickRoell-START-Hack-2026" / "UUID_helpers"
    if legacy_root.exists():
        return legacy_root

    return local_root


def _parse_channel_map(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"\{\s*en:\s*`([^`]+)`\s*,\s*_id:\s*`([^`]+)`\s*,?\s*\}", re.S)
    items = {}
    for name, uuid in pattern.findall(text):
        items[_normalize(name)] = {
            "category": "channel",
            "name": name,
            "uuid": uuid,
        }
    return items


def _parse_result_types(path: Path) -> dict[str, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\{\s*id:\s*(\d+),\s*name:\s*\"([^\"]+)\",\s*masterTestPrograms:\s*\[(.*?)\],\s*uuid:\s*\"([^\"]+)\",\s*unitTableIds:\s*\[(.*?)\],\s*\}",
        re.S,
    )

    items: dict[str, list[dict[str, Any]]] = {}
    for internal_id, name, programs_blob, uuid, units_blob in pattern.findall(text):
        programs = re.findall(r"\"([^\"]+)\"", programs_blob)
        units = re.findall(r"\"([^\"]+)\"", units_blob)
        key = _normalize(name)
        items.setdefault(key, []).append(
            {
                "category": "result",
                "id": int(internal_id),
                "name": name,
                "uuid": uuid,
                "programs": programs,
                "unit_table_ids": units,
            }
        )
    return items


def _parse_parameter_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = {}
    for row in data:
        name = row["en"]
        items[_normalize(name)] = {
            "category": "test_parameter",
            "name": name,
            "id": row["_id"],
        }
    return items


@lru_cache
def load_semantic_dictionary(helper_root: str | None = None) -> SemanticDictionary:
    root = Path(helper_root) if helper_root else _default_helper_root()

    channels = _parse_channel_map(root / "channelParameterMap.ts")
    results = _parse_result_types(root / "testResultTypes.ts")
    params = _parse_parameter_map(root / "TestParameterMap.json")

    return SemanticDictionary(
        channels_by_name=channels,
        results_by_name=results,
        params_by_name=params,
    )


def resolve_user_term(term: str, limit: int = 8) -> list[dict[str, Any]]:
    dictionary = load_semantic_dictionary()
    query = _normalize(term)

    candidates: list[dict[str, Any]] = []

    for key, value in dictionary.channels_by_name.items():
        if query in key or key in query:
            candidates.append(value)

    for key, value_list in dictionary.results_by_name.items():
        if query in key or key in query:
            candidates.extend(value_list)

    for key, value in dictionary.params_by_name.items():
        if query in key or key in query:
            candidates.append(value)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        unique_id = str(item.get("uuid") or item.get("id") or item.get("name"))
        key = (item["category"], unique_id)
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped[:limit]

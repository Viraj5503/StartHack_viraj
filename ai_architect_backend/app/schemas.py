from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentCategory(str, Enum):
    validation_compliance = "validation_compliance"
    comparison = "comparison"
    trend_drift = "trend_drift"
    hypothesis = "hypothesis"
    anomaly_check = "anomaly_check"
    data_selection = "data_selection"
    summary = "summary"


class CollectionName(str, Enum):
    tests = "Tests"
    values = "Values"


class PlanOperation(str, Enum):
    filtering = "filtering"
    aggregation = "aggregation"
    grouping = "grouping"
    time_series_extract = "time_series_extract"
    statistics = "statistics"
    compliance_check = "compliance_check"
    anomaly_scan = "anomaly_scan"
    hypothesis_probe = "hypothesis_probe"


class StatMethod(str, Enum):
    descriptive = "descriptive"
    t_test = "t_test"
    mann_whitney = "mann_whitney"
    rolling_mean = "rolling_mean"
    linear_slope = "linear_slope"
    z_score = "z_score"
    change_point_heuristic = "change_point_heuristic"
    threshold_check = "threshold_check"


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: Literal[
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
    ]
    value: Any


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_question: str
    normalized_question: str
    intent: IntentCategory
    required_collections: list[CollectionName]
    fields_needed: list[str] = Field(default_factory=list)
    operations: list[PlanOperation] = Field(default_factory=list)
    statistics_methods: list[StatMethod] = Field(default_factory=list)
    chart_needed: bool = False
    assumptions: list[str] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    follow_up_focus: list[str] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class PlannerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3)
    context: dict[str, Any] = Field(default_factory=dict)


class PlannerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: InvestigationPlan
    semantic_candidates: list[dict[str, Any]] = Field(default_factory=list)


class MongoQueryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: CollectionName
    pipeline: list[dict[str, Any]]
    explanation: str
    expected_shape: list[str] = Field(default_factory=list)


class QueryRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: InvestigationPlan
    max_repairs: int | None = Field(default=None, ge=0, le=5)
    semantic_candidates: list[dict[str, Any]] = Field(default_factory=list)


class QueryAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int
    pipeline: list[dict[str, Any]]
    error: str | None = None
    corrected_from_previous: bool = False


class QueryRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "failed"]
    candidate: MongoQueryCandidate
    attempts: list[QueryAttempt]
    row_count: int
    rows: list[dict[str, Any]]
    corrected_automatically: bool


class InsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: InvestigationPlan
    rows: list[dict[str, Any]] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class InsightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_3_sentences: list[str]
    anomaly_notes: list[str]
    recommendation: str
    follow_up_questions: list[str]
    chart_config: dict[str, Any]
    audit_log: list[str] = Field(default_factory=list)

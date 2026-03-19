from fastapi import FastAPI

from app.config import get_settings
from app.schemas import (
    InsightRequest,
    InsightResponse,
    PlannerRequest,
    PlannerResponse,
    QueryRunRequest,
    QueryRunResponse,
)
from app.services.insight import build_insight
from app.services.mongo_executor import MongoExecutor
from app.services.planner import build_plan

settings = get_settings()
app = FastAPI(title=settings.app_name)
executor = MongoExecutor()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "planner_mode": settings.planner_mode,
        "query_mode": settings.query_mode,
        "insight_mode": settings.insight_mode,
        "llm_provider": settings.llm_provider,
        "anthropic_ready": "yes" if bool(settings.anthropic_api_key) else "no",
        "openai_ready": "yes" if bool(settings.openai_api_key) else "no",
        "llm_ready": "yes"
        if (
            (settings.llm_provider.lower() == "anthropic" and bool(settings.anthropic_api_key))
            or (settings.llm_provider.lower() == "openai" and bool(settings.openai_api_key))
        )
        else "no",
    }


@app.post("/planner/plan", response_model=PlannerResponse)
def planner_plan(payload: PlannerRequest) -> PlannerResponse:
    plan, semantic_candidates = build_plan(payload.question, payload.context)
    return PlannerResponse(plan=plan, semantic_candidates=semantic_candidates)


@app.post("/query/run", response_model=QueryRunResponse)
def query_run(payload: QueryRunRequest) -> QueryRunResponse:
    return executor.run_plan_with_repair(payload.plan, payload.max_repairs, semantic_candidates=payload.semantic_candidates or None)


@app.post("/insight/generate", response_model=InsightResponse)
def insight_generate(payload: InsightRequest) -> InsightResponse:
    return build_insight(payload.plan, payload.rows, payload.stats)

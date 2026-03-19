from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE_PATH = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "materials-copilot-ai-architect"
    app_env: str = "dev"

    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str | None = None
    mongo_collection_tests: str = "_tests"
    mongo_collection_values: str = "valuecolumns_migrated"

    planner_mode: str = "mock"
    query_mode: str = "mock"
    insight_mode: str = "mock"
    llm_provider: str = "anthropic"

    anthropic_api_key: str | None = None
    anthropic_model_planner: str = "claude-sonnet-4-6"
    anthropic_model_query: str = "claude-sonnet-4-6"
    anthropic_model_insight: str = "claude-sonnet-4-6"

    openai_api_key: str | None = None
    openai_model_planner: str = "gpt-4o-mini"
    openai_model_query: str = "gpt-4o-mini"
    openai_model_insight: str = "gpt-4o-mini"

    max_query_repairs: int = 2
    max_query_rows: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()

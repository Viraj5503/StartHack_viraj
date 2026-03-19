# AI Architect Backend

This folder is your independent project workspace for the AI Architect role.

## Quick Start

From the root `StartHack_viraj` folder:

1. Run smoke test:
   - `./run_ai_architect.sh smoke`
2. Start API:
   - `./run_ai_architect.sh serve`
3. Run smoke test + API together:
   - `./run_ai_architect.sh all`
4. Evaluate planner quality:
   - `./run_ai_architect.sh eval`
   - default runs deterministic planner for speed
   - to evaluate with live Claude calls: `EVAL_PLANNER_MODE=llm ./run_ai_architect.sh eval`
5. Run end-to-end dry run:
   - `./run_ai_architect.sh dryrun`
   - or with custom question: `QUESTION="Show only Tuesday tests" ./run_ai_architect.sh dryrun`

## Claude Setup

1. Open `.env` in this folder and set `ANTHROPIC_API_KEY` locally.
2. Keep `PLANNER_MODE=llm` and `QUERY_MODE=llm` to enable Claude-backed planning and query repair.
3. Run planner benchmark:
   - `./run_ai_architect.sh eval`
4. Run end-to-end dry run:
   - `./run_ai_architect.sh dryrun`

If the API key is missing or Claude output is invalid, the backend automatically falls back to deterministic logic.

## Official Challenge MongoDB Setup

The challenge image is private on GHCR, so you need a GitHub PAT with `read:packages`.

From root `StartHack_viraj`:

1. Authenticate to GHCR:
   - `GHCR_USER=<your_github_username> GHCR_TOKEN=<your_pat> ./connect_challenge_mongo.sh login`
2. Pull image:
   - `./connect_challenge_mongo.sh pull`
3. Start container on 27018 (avoids conflict with local mongod on 27017):
   - `HOST_PORT=27018 ./connect_challenge_mongo.sh start`
4. Smoke test challenge Mongo:
   - `HOST_PORT=27018 ./connect_challenge_mongo.sh smoke`
5. Update backend `.env`:
   - `MONGO_URI=mongodb://localhost:27018`
   - optionally set `MONGO_DB_NAME=<database_name_from_smoke_output>`

## Endpoints

- `GET /health`
- `POST /planner/plan`
- `POST /query/run`
- `POST /insight/generate`

## Current Modes

Current default local `.env` setup:

- `PLANNER_MODE=llm`
- `QUERY_MODE=llm`
- `INSIGHT_MODE=mock`
- `LLM_PROVIDER=anthropic`

## Next Build Steps

1. Add LLM-powered insight mode once teammate stats payload is stable.
2. Add persistent query run logging for auditable traces.
3. Improve pipeline generation prompt with real collection schema samples from challenge Mongo.
4. Add prompt-level regression tests for query repair edge cases.

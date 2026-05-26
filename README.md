# Table Line Item Local Service (Windows EC2 / VS Code)

Standalone FastAPI service for chatting with uploaded documents using:

- AWS Bedrock models only
- local filesystem documents
- SQLite persistence in working directory

No runtime dependency on S3, SQS, Redis/Valkey, RDS/Postgres, or Secrets Manager.

## What this service provides

- `POST /v1/documents/upload`  
  Upload local files through Swagger, store metadata in SQLite, and index chunks.
- `POST /v1/analytics/table-line-item`  
  SSE chat endpoint with retrieval and optional spreadsheet SQL path.
- `GET /health`  
  Service health and runtime config.

## Supported file types

- `.txt`, `.md`, `.csv`, `.xlsx`, `.xls`, `.pdf`, `.docx`

## Local project structure

```text
table-line-item-local-service/
├── app/
│   ├── api/routes.py
│   ├── connectors/bedrock.py
│   ├── models/api_models.py
│   ├── storage/sqlite_store.py
│   ├── storage/document_indexer.py
│   ├── workflow/chat_state.py
│   ├── workflow/chat_prompts.py
│   ├── workflow/chat_nodes.py
│   ├── workflow/search_client.py
│   ├── workflow/table_sql_pipeline.py
│   ├── workflow/chat_workflow.py
│   └── main.py
├── scripts/run-local.ps1
├── scripts/smoke-test.ps1
├── requirements.txt
└── .env.example
```

## Setup on Windows EC2

1. Open PowerShell in `table-line-item-local-service`
2. Copy env template:

```powershell
copy .env.example .env
```

3. Ensure Bedrock credentials are available in your environment (for example via AWS profile/role).
4. Start:

```powershell
.\scripts\run-local.ps1
```

5. Open Swagger:

`http://localhost:8080/docs`

## Swagger workflow

1. Call `POST /v1/documents/upload` with:
   - `job_id`
   - `run_id`
   - `file`
2. Copy returned `file_id`.
3. Call `POST /v1/analytics/table-line-item` with JSON:

```json
{
  "job_id": "demo-job",
  "run_id": "run-001",
  "user_query": "Summarize this document.",
  "top_k": 5,
  "document_ids": ["<file_id>"]
}
```

You will receive server-sent events (`step`, `token`, `done`).

## SQLite persistence

Default DB: `./table_line_item.db`

Tables:

- `workflow_runs`
- `workflow_events`
- `conversation_checkpoints`
- `documents`
- `retrieval_chunks`
- `retrieval_queries`

## Smoke test

```powershell
.\scripts\smoke-test.ps1 -BaseUrl "http://localhost:8080" -FilePath "C:\path\to\sample.pdf"
```


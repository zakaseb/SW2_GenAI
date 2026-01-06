# CodeLlama C Refactorer

This fork repurposes the original requirement-processing app into a Streamlit-based C/C++ refactoring workspace. Upload `.c` or `.h` files, and the app will rewrite the code into a single, behaviorally equivalent but textually untraceable C source file using a CodeLlama backend. The legacy requirement-generation flow is disabled; downloads now return the rewritten `.c` output.

Note: Many legacy sections below describe the previous requirement-intelligence workflow and are kept for historical context.

---

## Table of Contents
- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [Architecture & Data Flow](#architecture--data-flow)
- [Inputs & Outputs](#inputs--outputs)
- [Processing Pipeline](#processing-pipeline)
- [Codebase Tour](#codebase-tour)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Stack](#running-the-stack)
- [Usage Walkthrough](#usage-walkthrough)
- [Data & Storage Layout](#data--storage-layout)
- [Background Requirement Jobs](#background-requirement-jobs)
- [Testing & Tooling](#testing--tooling)
- [Troubleshooting & Tips](#troubleshooting--tips)

---

## Overview

The application ships with:

- A **Streamlit UI** (`rag_deep.py`) for uploading engineering documents, loading verification context, and chatting with an LLM.
- An **Ollama-backed LangChain stack** (see `core/model_loader.py`) that provides both dense embeddings and a text-generation model.
- A **Docling-powered ingestion pipeline** (`core/document_processing.py`) that converts technical PDFs/DOCX into chunked LangChain `Document` objects and classifies each chunk as "General Context" vs. "Requirements".
- A **background requirement generation service** (`core/requirement_jobs.py`) that turns requirement chunks plus verification references into structured JSON and Excel outputs suitable for Jama.
- A **SQLite-backed auth and session layer** (`core/auth.py`, `core/database.py`, `core/session_manager.py`) so each engineer logs in, resumes prior work, and stores generated artifacts under `document_store/`.
- An **optional FastAPI wrapper** (`core/api.py`) that proxies Ollama calls and captures UI telemetry for debugging or proxy capture tools such as Postman (`Run_steps.txt` documents this workflow).

---

## Core Capabilities

- **Multi-format ingestion**: Upload single or multiple PDF, DOCX, or TXT files. Files are persisted under `document_store/pdfs/` and parsed with pdfplumber/python-docx before Docling converts them into sentence-aware chunks.
- **Automatic chunk classification**: Every chunk is labelled as general context or requirement text using `core.generation.classify_chunk`, allowing the UI to index them into separate in-memory vector stores (`GENERAL_VECTOR_DB` vs `DOCUMENT_VECTOR_DB`).
- **Layered retrieval context**: The UI can blend three knowledge sources—global verification methods (auto-loaded from `Verification Methods.docx`), per-user context uploads (`document_store/context_pdfs/`), and requirement chunks from the current session.
- **Conversational analysis**: Once documents are processed, users chat with the stack. The chat prompt injects relevant chunks, the last `MAX_HISTORY_TURNS` interactions, and persistent memory to keep follow-up questions grounded.
- **Jama-ready requirement jobs**: Clicking **Generate Requirements** streams requirement chunks, verification passages, and BM25-ranked general context into `core.requirement_jobs`. A background thread produces JSON + Excel deliverables stored in `document_store/generated_requirements/<user_id>/`.
- **Session persistence & auditing**: Login state is stored in `users.db`, and `package_session_for_storage` captures chat history, processed chunks, and downloads so a later login can rehydrate vector stores automatically (`re_index_documents_from_session`).
- **Operational telemetry**: With `USE_API_WRAPPER=True`, the UI posts structured `ui-event` payloads to the FastAPI wrapper so API logs—and, by extension, Postman capture sessions—show every user action without coupling the Streamlit process directly to Ollama.
- **Container-friendly deployment**: The provided `Dockerfile` plus `docker-compose.yml` run the Streamlit UI in a non-root container, mount the document store for persistence, and expose configuration knobs for pointing at a host Ollama instance.

---

## Architecture & Data Flow

```
┌──────────────┐      uploads/login      ┌─────────────┐       REST (optional)       ┌─────────────┐
│ Streamlit UI ├────────────────────────►│ FastAPI API ├────────────────────────────►│   Ollama    │
│  rag_deep.py │◄── session + jobs ──────┤ core/api.py │                              │ (LLM+emb)  │
└──────┬───────┘                          └─────────────┘                              └─────┬───────┘
       │     vector ops / doc chunks                 ▲                                       │
       │                                            logs                                     │
       ▼                                                                                    ▼
┌─────────────────────┐    chunk classify/index   ┌────────────────────┐   background jobs ┌────────────────────────┐
│  core.document_*    │──────────────────────────►│ InMemoryVectorStore│◄──────────────────│ core.requirement_jobs  │
│ Docling, BM25, etc.│                           │ (general/context/req)│                 │ Excel/JSON writers     │
└─────────────────────┘                           └────────────────────┘                   └──────────┬─────────────┘
       │                                                                                             │
       ▼                                                                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        document_store/ (pdfs, context_pdfs, generated_requirements)    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Inputs & Outputs

| Item | Source / Trigger | Format & Location | Notes |
| ---- | ---------------- | ----------------- | ----- |
| Project documents | User upload via main page | PDF, DOCX, TXT in `document_store/pdfs/` | Parsed by pdfplumber/python-docx/Text IO. |
| Context document | Sidebar upload or auto-copied `Verification Methods.docx` | Stored in `document_store/context_pdfs/` | Indexed into a persistent context vector store and used for every session when enabled. |
| User credentials | `manage_users.py` CLI → `users.db` | Username/password hash | Required before the UI unlocks ingestion tools. |
| Chat prompts | Streamlit chat input | Plain text (per session) | Stored in `st.session_state.messages` and persisted on logout. |
| Requirement jobs | "Generate Requirements" button | JSON + Excel saved under `document_store/generated_requirements/<user_id>/<job_id>.{json,xlsx}` | Excel schema matches the Jama import template described in `core.config.REQUIREMENT_JSON_PROMPT_TEMPLATE`. |
| Telemetry | UI actions | POST `/ui-event` payloads (FastAPI) | Used for auditing, proxy capture, or debugging. |

---

## Processing Pipeline

1. **Authentication & session bootstrap**
   - `core/auth.show_login_form` checks `users.db` (SQLite) via `core.database`.
   - Upon success the previous session snapshot is pulled (`load_session`), unpacked, and vector stores are rehydrated with `re_index_documents_from_session`.
   - Default verification context (`Verification Methods.docx`) is copied into `document_store/context_pdfs/` via `ensure_global_context_bootstrap` and indexed once per session.

2. **Document ingestion**
   - Uploaded files are saved under `PDF_STORAGE_PATH` using `save_uploaded_file`.
   - `load_document` expands them into LangChain documents; Docling’s `HybridChunker` breaks them into deduplicated chunks.
   - `classify_chunk` (LLM prompt) routes each chunk to either a general-context vector DB or the requirement vector DB. Requirements chunks also feed a BM25 corpus for lexical fallback.

3. **Context layering**
   - `core/search_pipeline` exposes helpers that load every chunk currently living in the context, general, or requirements vector stores.
   - Persistent context (verification methods) and sidebar uploads are indexed into `CONTEXT_VECTOR_DB` and mirrored into `PERSISTENT_VECTOR_DB` so they survive resets.

4. **Conversational retrieval & answer generation**
   - The chat prompt bundles: (a) last `MAX_HISTORY_TURNS` user/assistant messages, (b) persistent "memory" (used to preserve long-running context), (c) concatenated chunks from all vector stores.
   - Generation flows through LangChain’s `ChatPromptTemplate` defined in `GENERAL_QA_PROMPT_TEMPLATE`.

5. **Requirement-generation background jobs**
   - `core.requirement_jobs.RequirementJobManager` enqueues a thread that loops over requirement chunks, retrieves BM25-ranked general context, concatenates verification references, and calls `generate_requirements_json`.
   - Parsed JSON is turned into a Pandas DataFrame and exported via `generate_excel_file`. Results plus metadata are recorded in `requirement_jobs` within SQLite for progress tracking.

---

## Codebase Tour

| Path | Description |
| ---- | ----------- |
| `rag_deep.py` | Streamlit entry point: login flow, file upload, context management, chat UI, and requirement job button wiring. |
| `core/config.py` | Central constants, prompt templates, directory defaults, and endpoint URLs. Supports overriding via `core/config.local.json` or environment variables. |
| `core/model_loader.py` | Lazy-loads Ollama embedding & generation models plus the `sentence-transformers` CrossEncoder reranker (download helper: `pre_download_model.py`). |
| `core/document_processing.py` | File persistence, parsing via pdfplumber / python-docx, Docling chunking, chunk classification, and vector-store indexing helpers. |
| `core/generation.py` | Prompt chains for Q&A, requirement extraction, summarization, keyword extraction (utility), chunk classification, and Excel export utilities. |
| `core/search_pipeline.py` | Convenience functions to pull all currently indexed chunks from each vector store. |
| `core/session_manager.py` & `core/session_utils.py` | Session initialization, reset logic, vector store rehydration, uploader resets, and persistent session packaging. |
| `core/database.py` | SQLite schema + helpers for users, stored sessions, and requirement job metadata. |
| `core/requirement_jobs.py` | Background job manager, BM25 ranking of general context, JSON parsing, and Excel writing for Jama imports. |
| `core/api.py` | FastAPI wrapper that proxies `/generate`, `/embed`, and `/ui-event` calls to Ollama, adding structured logging for every HTTP exchange. |
| `manage_users.py` | CLI for initializing the DB, adding users, and listing existing accounts. |
| `pre_download_model.py` | One-off script to eagerly download and cache the CrossEncoder reranker to avoid first-run latency. |
| `tests/` | Pytest suite covering config defaults, doc processing, search helpers, session management, and RAG orchestration. |

---

## Installation

### 1. Prerequisites

- **Python** 3.9–3.12 (the Dockerfile pins 3.9; local virtualenvs work with 3.12 as well).
- **Ollama** running on the same machine or reachable over the network. Pull the models you plan to reference (defaults: `mistral:7b` for both embeddings and generation).
- **Git**, **pip**, and build-essential libraries required by Docling (Poppler, libGL, etc., depending on your OS).
- Optional: **Docker**/**Docker Compose** for container deployment, **Postman** if you plan to capture API traffic per `Run_steps.txt`.

### 2. Clone the repository

```bash
git clone https://github.com/chintanboghara/DocuMind-AI.git
cd DocuMind-AI
```

### 3. Create & activate a virtual environment

**Windows (PowerShell)**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

Choose the requirement file that fits your environment:

- **Full stack (Docling, GPU-friendly)**:
  ```bash
  pip install -r requirements_sw_final.txt
  ```
- **Slim stack / legacy environments** (no Docling, fewer extras):
  ```bash
  pip install -r requirements.txt
  ```

If you install the slim stack, add Docling manually:
```bash
pip install docling docling-core docling-ibm-models docling-parse
```

### 5. Initialize the user database & create accounts

```bash
python manage_users.py init
python manage_users.py add   # prompts for username and password
python manage_users.py list  # optional sanity check
```

### 6. Pre-download the reranker (optional but recommended)

```bash
python pre_download_model.py
```

This caches `cross-encoder/ms-marco-MiniLM-L-6-v2` so the first chat run does not block on model downloads.

---

## Configuration

All configuration values live in `core/config.py`. You can override them via:

1. Environment variables (export before launching Streamlit or set them in `.env` / Docker Compose).
2. A JSON override file at `core/config.local.json` (same keys as `core/config.py`).

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `OLLAMA_BASE_URL` | Where LangChain should call Ollama for embeddings & generation. | `http://127.0.0.1:11434` |
| `OLLAMA_EMBEDDING_MODEL_NAME` | Ollama model tag used for embeddings. Must support `/api/embeddings`. | `mistral:7b` |
| `OLLAMA_LLM_NAME` | Ollama model used for answer & requirement generation. | `mistral:7b` |
| `RERANKER_MODEL_NAME` | `sentence-transformers` CrossEncoder checkpoint. | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `PDF_STORAGE_PATH` | Root folder for uploaded working files. | `document_store/pdfs` |
| `CONTEXT_PDF_STORAGE_PATH` | Folder for persistent context docs (`Verification Methods.docx`, sidebar uploads). | `document_store/context_pdfs` |
| `MEMORY_FILE_PATH` | Reserved path for long-term memory artifacts (not currently used). | `document_store/memory/context.json` |
| `REQUIREMENTS_OUTPUT_PATH` | Base folder for generated Excel/JSON per user. | `document_store/generated_requirements` |
| `RAG_API_BASE` | Base URL of the FastAPI wrapper (`core/api.py`). | `http://127.0.0.1:8000` |
| `USE_API_WRAPPER` | When `True`, the UI posts `/ui-event` telemetry to the wrapper. | `True` |
| `RAG_API_KEY` | Optional auth token if you gate the wrapper. | `""` |
| `POSTMAN_PROXY` | Set to `http://127.0.0.1:<port>` to route wrapper→Ollama calls through a proxy such as Postman. | unset |
| `LOG_LEVEL` | Logging level consumed by `core/logger_config.py`. | `INFO` |

---

## Running the Stack

### Local (direct mode)

1. **Start Ollama** (terminal #1):
   ```bash
   ollama serve
   ```
2. **Activate your virtualenv** (terminal #2) and export any overrides:
   ```bash
   cd /path/to/DocuMind-AI
   source venv/bin/activate
   export OLLAMA_BASE_URL=http://127.0.0.1:11434
   export LOG_LEVEL=INFO
   ```
3. **Run Streamlit**:
   ```bash
   streamlit run rag_deep.py
   ```
4. Open the provided `http://localhost:8501` URL, log in, and start uploading documents.

### Local with FastAPI wrapper & Postman capture (optional)

`Run_steps.txt` contains a step-by-step playbook for:

1. Launching Postman in proxy capture mode (default port `5560`).
2. Setting `HTTP_PROXY` / `HTTPS_PROXY` in both the FastAPI and Streamlit shells so all Ollama-bound traffic flows through Postman.
3. Starting the FastAPI wrapper:
   ```bash
   uvicorn core.api:app --host 0.0.0.0 --port 8000
   ```
4. Running Streamlit with `NO_PROXY=huggingface.co` to keep Hugging Face downloads out of the proxy.

This workflow is useful when you need to audit every `/generate`/`/embed` call without instrumenting Ollama directly.

### Docker Compose

1. Ensure Ollama is running on the host (or amend `docker-compose.yml` to point to your Ollama container/network).
2. From the repo root:
   ```bash
   docker-compose up --build -d
   ```
3. Visit `http://localhost:8501`. Uploaded files and generated artifacts live in the host `document_store/` folder because it is bind-mounted into the container.
4. Stop the stack with:
   ```bash
   docker-compose down
   ```

---

## Usage Walkthrough

1. **Log in** with a user created via `manage_users.py`. Sessions persist between logins.
2. **(Optional) Upload a context document** from the sidebar. Successful uploads are indexed into the persistent context store, and the UI displays a success toast.
3. **Upload project documents** via the main file uploader. Multiple files are supported, and re-uploading a different set automatically resets chat + vector stores to avoid stale context.
4. **Review processing status**: Streamlit shows counts of general vs requirement chunks and lists every file that was successfully parsed.
5. **Chat with the assistant**: Once `document_processed` is true, the chat input becomes active. Ask natural-language questions; responses include verification context, general context, and requirement chunks.
6. **Manage memory**: Use sidebar controls to clear chat, delete the context document (`purge_persistent_memory`), or reset documents, which also clears vector stores and file-upload state.
7. **Generate requirements**: After at least one requirement chunk exists, click **Generate Requirements**. The sidebar reports job status (queued/running/completed/failed) and exposes a download button when the Excel file is ready. The 3 most recent jobs are listed for quick auditing.
8. **Download or inspect outputs**: Completed jobs offer both a downloadable Excel and a JSON preview rendered in the sidebar text areas.
9. **Logout**: When logging out, the session snapshot is saved (`save_session`), ensuring you can resume later without re-uploading documents.

---

## Data & Storage Layout

```
document_store/
├── pdfs/                    # working copies of uploaded project docs
├── context_pdfs/            # persistent verification/context uploads (auto-seeded with Verification Methods.docx)
├── generated_requirements/  # per-user folders containing <job_id>.json and <job_id>.xlsx
└── memory/
    └── context.json         # reserved for future long-term memory features
```

Additional sample documents live under `SW_PoC/` and `dummy20250710102410/` for testing or demonstrations.

---

## Background Requirement Jobs

- Jobs are keyed by UUID (`requirement_jobs.id`) and run in a thread pool (`ThreadPoolExecutor`) capped at two concurrent workers.
- Each chunk is enriched with:
  - The entire verification document (joined with `---` separators).
  - Up to `top_k_general` BM25-ranked general-context paragraphs (default 8) to reinforce domain knowledge, with an optional character cap (`safe_char_cap`, default 32 k) to keep prompts manageable.
- Responses from the LLM are parsed by `parse_requirements_payload`, cleaned, and validated before export.
- Metadata tracks the number of requirements produced, char counts, and file paths for JSON/Excel so the UI can surface progress and download links.
- Failures (e.g., no usable JSON) update the job row with `status="failed"` and an error message, which is surfaced in the sidebar.

---

## Testing & Tooling

- **Unit tests**: Run `pytest` from the repo root (virtualenv activated).
  ```bash
  pytest
  ```
  The suite covers configuration defaults, document processing edge cases, search helpers, session management, and rag_deep orchestration.
- **User management**: `manage_users.py` (see [Installation](#installation)).
- **Model warm-up**: `python pre_download_model.py`.
- **Logging**: Adjust `LOG_LEVEL` or inspect `streamlit.log` and the FastAPI console for detailed traces.

---

## Troubleshooting & Tips

- **Ollama connection errors**: Ensure `OLLAMA_BASE_URL` is reachable and that the chosen model supports both `/api/generate` and `/api/embeddings`. The UI fails fast with a critical error if either model fails to load.
- **Docling dependencies**: On Linux you may need system packages such as `libgl1` and `poppler-utils`. Install them before running `pip install -r requirements_sw_final.txt`.
- **Large file processing**: Chunking large PDFs can exhaust memory. Monitor the console for `MemoryError` warnings logged by `load_document` and consider splitting documents.
- **Proxy capture issues**: When routing through Postman, set `NO_PROXY=huggingface.co` so model downloads from Hugging Face bypass the proxy (per `Run_steps.txt`).
- **Persisted context confusion**: After purging context documents, `allow_global_context` is disabled for the session. Re-enable it by uploading a new context doc or toggling the flag in `st.session_state`.
- **Requirement job stuck in queued/running**: Check `users.db` → `requirement_jobs` table for error messages, and inspect `document_store/generated_requirements/<user_id>/` for partially written files.
- **Docker networking**: On Linux, `host.docker.internal` may not resolve. Set `OLLAMA_BASE_URL=http://172.17.0.1:11434` (or your actual bridge IP) in `docker-compose.yml`.

---

DocuMind-AI is designed to be extended—whether that means swapping in a different LLM via Ollama, connecting the FastAPI wrapper to additional observability tooling, or wiring the existing summarization/keyword utilities into the UI. Use this README as your checklist for understanding the moving parts before you customize them.



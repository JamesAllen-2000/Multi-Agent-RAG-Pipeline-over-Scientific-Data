# Multi-Agent RAG Pipeline over Scientific Data (MVP)

A minimal multi-agent retrieval and reasoning pipeline that answers research questions using **multiple source types**: **PDFs/papers** (local files or **arXiv API**), **datasets** (local CSV or **Kaggle** via kagglehub). A **retrieval agent** plans where to search, a **reasoning agent** can use a **calculator tool**, and a **query API** returns the answer with **sources and latency breakdown**.

## Level 1 - Core (This MVP)

| Requirement | Implementation |
|-------------|----------------|
| Ingestion (>=2 source types) | **Document**: PDF/txt -> chunk -> OpenAI embeddings -> Chroma. **Structured**: CSV -> metadata + path stored; queried via pandas. **arXiv**: [arXiv API](https://info.arxiv.org/help/api/user-manual.html) (search_query / id_list) -> metadata + title+abstract in Chroma. **Kaggle**: [kagglehub](https://github.com/Kaggle/kagglehub) `dataset_download(slug)` -> ingest CSVs as structured, PDF/txt/md as document (with `dataset_slug` in metadata). |
| Retrieval agent | **Planner**: one LLM call (Groq or OpenAI) that outputs a JSON plan (which source type + query per step). **Executor**: deterministic vector search (Chroma), structured table read (pandas), or **live arXiv API** call (search_query -> Atom -> title+abstract chunks). |
| Reasoning agent + >=1 tool beyond retrieval | **Reasoning**: one LLM call with retrieved evidence; must cite sources. **Tool**: `calculator(expression)` - safe arithmetic only. |
| Query API | **POST /query** returns `answer`, `sources`, `latency` (planning_ms, retrieval_ms, reasoning_ms, total_ms). |

## Explainability And LangChain

- We use LangChain for specific structural benefits while keeping the overall orchestration transparent so there's "no magic."
- **LangChain Abstractions Used**:
  - `ChatOpenAI`: A unified chat model interface used for both OpenAI and Groq APIs.
  - `.with_structured_output()`: Ensures our Planner LLM outputs precisely the defined Pydantic `RetrievalPlan` schema dynamically, avoiding the need for flaky regex or manual JSON parsing retries.
  - `.bind_tools()` & `ToolMessage`: Used in the reasoning agent to seamlessly expose the `calculator` tool to the LLM and cleanly pass the tool's execution results back into the message history array.
  - `HuggingFaceEmbeddings`: Used as a local offline embedding wrapper for `sentence-transformers/all-MiniLM-L6-v2`. It natively provides batch `.embed_documents()` to optimize metadata and vector ingestion speeds.
- Chroma is used purely as a raw vector store without LangChain wrapping (`upsert` and `query_embeddings`), allowing complete manual control over metadata tagging and distance scoring deduplication.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# For Groq (chat): set LLM_PROVIDER=groq and GROQ_API_KEY
# For OpenAI (chat): set LLM_PROVIDER=openai and OPENAI_API_KEY
# Embeddings: local HuggingFace sentence-transformers by default.
```

## Ingestion

```bash
python -m scripts.ingest document path/to/paper.pdf --title "My Paper"
python -m scripts.ingest structured path/to/dataset.csv --title "Measurements"
python -m scripts.ingest arxiv --query "ti:electron thermal conductivity" --max-results 20
python -m scripts.ingest arxiv --id-list "2301.00001,2302.00002"
python -m scripts.ingest kaggle --dataset "Cornell-University/arxiv"
```

Data is stored under `data/`: Chroma DB in `data/chroma`, SQLite metadata in `data/sources.db`, CSV copies in `data/tables/`.

## Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **POST /query** - body: `{"question": "What is the main finding of the paper?"}`
- **GET /health** - liveness
- **GET /ready** - readiness (API key, SQLite, Chroma)

## Project Layout

```
app/
  config.py
  logging_config.py
  health.py
  models.py
  db.py
  main.py
  llm_client.py
  langchain_client.py   # Optional LangChain ChatOpenAI wrapper
  arxiv_client.py
  ingestion/
  retrieval/
  reasoning/
scripts/
  ingest.py
```

## Level 2/3 Status

- Implemented now: bounded concurrency, source attribution, structured logs, readiness checks, data-version increments on ingestion.
- Not fully implemented yet: retrieval-result cache with invalidation, entity graph traversal, verification agent, systematic eval harness, hardened code-execution sandbox (calculator is restricted and non-general).

# Multi-Agent RAG Pipeline over Scientific Data

A minimal multi-agent retrieval and reasoning pipeline that answers research questions using heterogeneous sources. The system retrieves information across **papers/documents** (local PDFs or the live **arXiv API**) and **structured databases** (Kaggle datasets). It features a **retrieval planner agent** that decides where to search, and a **reasoning agent** that synthesizes evidence while optionally calling a calculator tool.

---

## 🚀 Setup, Configuration, and Execution

### Prerequisites
- Python 3.11 is recommended.
- You will need API keys for your preferred LLM provider (OpenAI or Groq) and optionally Kaggle limits.

### Installation

```bash
# 1. Create and activate a virtual environment
python -3.11 -m venv venv311
venv311\Scripts\activate   # Windows
# source venv311/bin/activate  # Linux/macOS

# 2. Install requirements
pip install -r requirements.txt

# 3. Setup Configuration
cp .env.example .env
```

### Environment Variables (`.env`)
The pipeline is driven entirely by environment variables. At a minimum, set your LLM provider and keys:

```env
# Choose 'groq' or 'openai' (Groq used for fast Llama3 reasoning)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here

# OpenAI key is REQUIRED if using OpenAI for text embeddings
OPENAI_API_KEY=your_openai_key_here

# Enable/disable LangChain abstractions
USE_LANGCHAIN=true

# For Kaggle dataset ingestion
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

### Running Ingestion
You can ingest multiple source types into the local Chroma and SQLite databases using the unified CLI:

```bash
# Ingest local PDFs or Data
python -m scripts.ingest document path/to/paper.pdf --title "My Paper"
python -m scripts.ingest structured path/to/dataset.csv --title "Measurements"

# Ingest live from arXiv
python -m scripts.ingest arxiv --query "ti:electron thermal conductivity" --max-results 20
python -m scripts.ingest arxiv --id-list "2301.00001,2302.00002"

# Ingest an entire Kaggle repository
python -m scripts.ingest kaggle --dataset "Cornell-University/arxiv"
```

### Running the API & Testing
Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- **POST `/query`** - Submit question: `{"question": "What is electron thermal conductivity?"}`
- **GET `/health`** - Basic liveness pin
- **GET `/ready`** - Checks API keys, SQLite connection, and Chroma database status.

**Local Testing Script**:
You can run `python test_api.py` to trigger local python end-to-end questions directly through the query pipeline without running the server, instantly outputting the answer, the number of cited sources, and a latency breakdown.

---

## ✨ Features Implemented

### Level 1 (Core)
- **Heterogeneous Ingestion**: Handled 4 separate sources smoothly: Documents (PDF/Txt via Chunking), Structured tabular datasets (CSV/Pandas), Live web endpoints (ArXiv Atom API feeds), and entire zip repositories (KaggleHub).
- **Retrieval Planner**: A dedicated agent that analyzes the question and outputs a structured JSON plan detailing which sources to query and what text to search for.
- **Reasoning Agent & Tools**: Synthesizes the retrieved contextual evidence, explicitly cites the required documents, and connects to a `calculator(expression)` tool allowing dynamic math evaluation on retrieved scientific data points.
- **Latency Tracking API**: Returning comprehensive metrics broken down into `planning_ms`, `retrieval_ms`, `reasoning_ms`.

### Level 2 (Scalability)
- **Concurrent Query Handling**: Achieved via FastAPI and standard `asyncio`. The pipeline wraps all tool calls, database operations, and HTTP requests in asyncio tasks / threading pools, ensuring concurrent API requests do not interfere or block each other.
- **Data Versioning Framework**: Integrated SQLite to track metadata, source origins, and dataset versions to prevent duplicate extraction on repeated ingestion.
- *(Limitation)*: The caching implementation does not feature intelligent graph traversal or smart prompt caching (currently fetches fresh queries every time).

### Level 3 (Robustness)
- **Abstention ("Confidently Wrong" Catching)**: The reasoning agent operates under the strict system prompt rule: *"Reasoning without grounding is hallucination."* If the retrieved evidence is poor or empty, it throws a fallback failure instead of attempting to blindly guess.
- **Sandboxed execution attempts**: Tool execution limits the calculator to safe arithmetic logic. 
- *(Limitation)*: The sandbox is native python arithmetic evaluation, not a secure container. We lack a large-scale evaluation harness (like Ragas) across 1000s of questions.

---

## 🏛️ Key Architectural Decisions

**FastAPI & Asyncio Orchestration**
Instead of a heavy Celery worker architecture, I chose FastAPI driven entirely by `asyncio` for the backend. Since the core bottleneck of a multi-agent system is I/O latency (waiting for multiple LLMs and APIs to respond), `asyncio.to_thread` and `asyncio.gather` allow massive concurrent query processing without interference, keeping the whole pipeline highly portable and lightweight.

**Vector Storage vs. Metadata Grounding (Chroma + SQLite)**
I explicitly separated document vector storage (ChromaDB) from ingestion metadata (SQLite). Chroma handles purely dumb similarity distance metrics on chunks. SQLite handles the relational aspect: Tracking which file or ArXiv URL generated which chunk, storing table shapes, and managing ingestion idempotency so the system won't index the same PDF twice.

**"No Magic" LangChain Integration**
LangChain is included but restricted to very specific functions rather than acting as a black-box orchestrator. I used `.with_structured_output()` to force correct JSON schema generation from the planner (which raw LLM calls frequently get wrong), and `.bind_tools()` to seamlessly expose the Python calculator function without writing manual string-to-function parsing mapping. The primary control loop, prompts, and tool results are handled explicitly in native Python logic.

**Agent Separation (Planner vs Reasoner)**
I chose a two-step agent model instead of one massive ReAct agent. The Planner *only* knows how to search, outputting JSON arrays. The Reasoner *only* knows how to synthesize and do math based on the given context. This architectural hard-split enforces the "no reasoning without grounding" philosophy and makes it incredibly easy to debug which step failed.

---

## 🤖 Interesting LLM Conversations

While implementing the integration between LangChain's structured JSON output and the Groq (Llama-3) API, the pipeline continually crashed with `400 Bad Request` exceptions originating directly from the Open AI SDK wrapper. The LLM debugging revealed that LangChain natively passes `strict=True` inside the JSON schema payload definition. Groq’s backend rejected this strict constraint schema structure. We resolved it by dynamically passing `method="json_mode"` locally for Groq, bypassing the strict validation while still successfully enforcing Pydantic models.

---

## ⏳ What I Would Do Differently With More Time

**Implement an Evaluation Harness**
I would build a systematic eval pipeline (using a framework like `Ragas` or `DeepEval`). Currently, validation requires manually reading the answers. With more time, I would automatically generate 500 scientific question-answer pairs against the ingested Kaggle dataset, run them through the retrieval and reasoning agents, and quantitatively score their "Context Precision", "Answer Faithfulness", and "Citation Recall" across layout adjustments to catch regressions when tweaking prompts or models.

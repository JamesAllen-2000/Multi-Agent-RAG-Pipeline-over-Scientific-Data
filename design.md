# design.md
# System Design and Architecture
## Multi-Agent RAG Pipeline over Scientific Data

---

## 1. Design Philosophy

The system is built around a single guiding principle:

**Reasoning without grounding is hallucination.**

To enforce this:
- Retrieval, reasoning, and verification are separated
- Deterministic components guard probabilistic ones
- The system prefers abstention over unsupported answers

---

## 2. High-Level Architecture

Client  
→ FastAPI API Layer  
→ Query Orchestrator  
→ Retrieval Planner (LLM)  
→ Retrieval Executors (Deterministic Tools)  
→ Reasoning Agent (LLM)  
→ Verification Agent (LLM)  
→ Response Assembler  

---

## 3. Storage Layer

### PostgreSQL
- Source metadata
- Provenance tracking
- Cache keys and versions
- Evaluation artifacts

### Vector Database (Qdrant or Chroma)
- Embedded document chunks
- Similarity search only

Structured datasets are not embedded and are queried directly.

---

## 4. Ingestion Pipeline

### Unstructured Documents
1. Parse PDF
2. Chunk text with overlap
3. Generate embeddings
4. Store embeddings in vector DB
5. Persist metadata in Postgres

### Structured Data
1. Validate schema
2. Register dataset
3. Store metadata and query interface
4. Enable deterministic querying (SQL / pandas)

---

## 5. Query Execution Flow

### Step 1: Retrieval Planning
An LLM determines:
- Which sources to consult
- What queries to issue per source

The output is strict JSON to enable validation and retry.

---

### Step 2: Retrieval Execution
Each retrieval action is:
- Deterministic
- Logged with latency
- Source-attributed

No LLM involvement at this stage.

---

### Step 3: Reasoning
The reasoning agent:
- Receives only retrieved evidence
- Cannot introduce new facts
- Must cite every claim

If evidence is insufficient, the agent must explicitly abstain.

---

### Step 4: Verification
A separate verifier agent:
- Re-evaluates all claims
- Checks citation support
- Flags unsupported assertions

Verifier failure results in:
- Warning surfaced to the user
- Optional retry with stricter constraints

---

## 6. Concurrency and Isolation

- Each query runs in its own asyncio task group
- No shared mutable state between queries
- Stateless agents, stateful orchestration

This prevents cross-query leakage and race conditions.

---

## 7. Caching Strategy

- Cache retrieval results, not final answers
- Cache key includes:
  - Normalized question hash
  - Global data version

On new ingestion:
- Increment data version
- Automatically invalidate stale cache entries

---

## 8. Failure Modes and Mitigations

| Failure Mode | Mitigation |
|-------------|-----------|
| Hallucinated reasoning | Verification agent |
| Partial retrieval | Explicit insufficiency response |
| High load | Bounded concurrency |
| Stale cache | Versioned invalidation |
| Tool misuse | Sandboxed execution |

---

## 9. Security Considerations

- Secrets via environment variables
- Optional code execution sandboxed
- No external network access for tools
- Strict input/output validation between agents

---

## 10. Trade-Offs

- Additional latency due to verification
- Increased complexity from agent separation
- Higher cost due to multiple LLM calls

These trade-offs are deliberate to favor **trustworthiness over speed**.

---

## 11. Future Improvements

- Larger automated evaluation suite
- Entity-relationship graph traversal
- Cross-document contradiction detection
- Adaptive retrieval planning based on past failures
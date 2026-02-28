import asyncio
from app.main import _run_pipeline
from app.config import get_settings
from app.models import QueryRequest

async def test_pipeline():
    settings = get_settings()
    
    queries = [
        "What is electron thermal conductivity in novel materials and what are the measurements?",
        "What is the test value from the dataset?"
    ]
    
    for q in queries:
        print(f"\n--- Testing Query: '{q}' ---")
        try:
            req = QueryRequest(question=q)
            resp = await _run_pipeline(req, settings)
            print("Answer:", resp.answer)
            print("Sources:", len(resp.sources))
            for s in resp.sources:
                print(f" - {s.source_type} ({s.source_id}): {s.excerpt[:100]}...")
            print("Latency:", resp.latency)
        except Exception as e:
            print("Pipeline failed:", e)

if __name__ == "__main__":
    asyncio.run(test_pipeline())

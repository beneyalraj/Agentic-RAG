import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas import QueryRequest, QueryResponse, HealthResponse
from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.retrieval.vector_store import PineconeVectorStore
from config.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):

    thread_id = request.thread_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "query": request.query,
        "messages": [],
        "tool_calls_made": [],
        "tool_results": [],
        "final_answer": None,
        "iteration_count": 0,
    }

    try:
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await agent_graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution error: {str(e)}"
        )

    answer = final_state.get("final_answer", "Sorry, I could not generate an answer.")
    tool_calls = final_state.get("tool_calls_made", [])

    return QueryResponse(
        answer=answer,
        tool_calls_made=tool_calls,
        thread_id=thread_id,
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    status = "ok"
    components = {}

    try:
        vs = PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        vs.create_index_if_not_exists()
        stats = vs.describe_index_stats()
        components["pinecone"] = {
            "status": "ok",
            "vector_count": stats.get("total_vector_count", 0),
        }
    except Exception as e:
        logger.warning(f"Pinecone health check failed: {e}")
        components["pinecone"] = {"status": "error", "detail": str(e)}
        status = "degraded"

    db_path = Path("data/financials.db")
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM financial_metrics LIMIT 1")
            cursor.fetchone()
            conn.close()
            components["financials_db"] = {"status": "ok", "path": str(db_path)}
        except Exception as e:
            components["financials_db"] = {"status": "error", "detail": str(e)}
            status = "degraded"
    else:
        components["financials_db"] = {"status": "error", "detail": "file not found"}
        status = "degraded"

    return HealthResponse(status=status, components=components)
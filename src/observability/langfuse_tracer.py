import logging
from contextlib import asynccontextmanager
from typing import Optional

from langfuse import Langfuse
from langfuse.decorators import langfuse_context

from config.config import settings

logger = logging.getLogger(__name__)

langfuse_client = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_BASE_URL,
)

def flush():

    langfuse_client.flush()
    logger.info("Langfuse traces flushed.")

@asynccontextmanager
async def trace_request(query: str, thread_id: str, user_id: str = "anonymous"):

    trace = langfuse_client.trace(
        name="agentic-rag-query",
        input={"query": query},
        metadata={
            "thread_id": thread_id,
            "user_id": user_id,
            "environment": "development",
        },
        tags=["agentic-rag", "financial"],
    )
    langfuse_context.update_current_trace(trace)

    try:
        yield trace

    finally:
        flush()
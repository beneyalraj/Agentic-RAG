from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question")
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID for continuity. If not provided, a new UUID is generated."
    )

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Final answer from the agent")
    tool_calls_made: List[str] = Field(..., description="List of tool names invoked")
    thread_id: str = Field(..., description="Thread ID used for this query")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status: 'ok' or 'degraded'")
    components: dict = Field(..., description="Per‑component health details")
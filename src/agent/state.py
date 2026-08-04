import operator
from typing import Annotated, List, Dict, Any, Optional, TypedDict

class AgentState(TypedDict):
    
    query: str
    messages: Annotated[List[Dict[str, Any]], operator.add]
    tool_calls_made: Annotated[List[str], operator.add]
    tool_results: Annotated[List[Dict[str, Any]], operator.add]
    
    pending_tool_calls: List[Dict[str, Any]]
    
    final_answer: Optional[str]
    iteration_count: int
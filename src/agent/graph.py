from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState
from src.agent.nodes import reasoning_node, tool_execution_node, finalizer_node
from config.config import settings

def should_continue(state: AgentState) -> str:

    if state.get("pending_tool_calls") and len(state["pending_tool_calls"]) > 0:
        return "tool_execution"
    return "finalizer"

def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.set_entry_point("reasoning")

    workflow.add_conditional_edges(
        "reasoning",
        should_continue,
        {
            "tool_execution": "tool_execution",
            "finalizer": "finalizer",
        }
    )

    workflow.add_edge("tool_execution", "reasoning")

    workflow.add_edge("finalizer", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

agent_graph = build_agent_graph()
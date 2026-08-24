import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import logging
from dotenv import load_dotenv
load_dotenv()

from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.observability.langfuse_tracer import trace_request, flush   # <-- import trace_request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    query = "What was JPMorgan's ROE using their most recent net income and stockholders equity?"
    thread_id = "test-langfuse"

    # Wrap the whole agent run in a trace
    async with trace_request(query, thread_id) as trace:
        initial_state: AgentState = {
            "query": query,
            "messages": [],
            "tool_calls_made": [],
            "tool_results": [],
            "final_answer": None,
            "iteration_count": 0,
        }
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await agent_graph.ainvoke(initial_state, config=config)
        logger.info(f"Final answer: {final_state.get('final_answer')}")

        # Optionally update the trace output
        trace.output = {"answer": final_state.get("final_answer")}

    # Flush traces after the context exits
    flush()
    logger.info("Done. Check your Langfuse dashboard.")

if __name__ == "__main__":
    asyncio.run(main())
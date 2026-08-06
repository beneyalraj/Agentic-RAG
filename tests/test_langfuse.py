import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import logging
from src.agent.graph import agent_graph
from src.agent.state import AgentState
from src.observability.langfuse_tracer import flush

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    query = "What was JPMorgan's ROE using their most recent net income and stockholders equity?"
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "tool_calls_made": [],
        "tool_results": [],
        "final_answer": None,
        "iteration_count": 0,
    }
    config = {"configurable": {"thread_id": "test-langfuse"}}
    final_state = await agent_graph.ainvoke(initial_state, config=config)
    logger.info(f"Final answer: {final_state.get('final_answer')}")
    logger.info("Flushing traces...")
    flush()
    logger.info("Done. Check your Langfuse dashboard.")

if __name__ == "__main__":
    asyncio.run(main())
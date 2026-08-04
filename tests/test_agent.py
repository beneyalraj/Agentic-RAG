import asyncio
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pprint import pprint
from src.agent.graph import agent_graph
from src.agent.state import AgentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

    logger.info(f"Running agent with query: {query}")
    config = {"configurable": {"thread_id": "test-1"}}

    final_state = None
    async for event in agent_graph.astream(initial_state, config=config):
        for node_name, node_output in event.items():
            logger.info(f"Node '{node_name}' completed.")
            final_state = node_output

            if node_name == "reasoning":
                msgs = node_output.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    if last.get("role") == "assistant":
                        if "tool_calls" in last:
                            logger.info(f"  -> Tool calls requested: {last['tool_calls']}")
                        else:
                            logger.info(f"  -> Assistant said: {last.get('content', '')[:200]}...")

            elif node_name == "tool_execution":
                results = node_output.get("tool_results", [])
                for r in results:
                    logger.info(f"  -> Tool '{r['tool_name']}' returned: {r['result'][:200]}...")

            elif node_name == "finalizer":
                final = node_output.get("final_answer")
                logger.info(f"  -> Final answer: {final}")

    print("FINAL ANSWER:")
    print(final_state.get("final_answer", "No final answer.") if final_state else "No output.")

if __name__ == "__main__":
    asyncio.run(main())
import sys
import types
from pydantic import v1 as pydantic_v1

_stub = types.ModuleType("langchain_community.chat_models.vertexai")

class _StubChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("VertexAI is not used in this project.")

_stub.ChatVertexAI = _StubChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub
sys.modules["langchain_core.pydantic_v1"] = pydantic_v1

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import json
import logging
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from tqdm import tqdm
from ragas.run_config import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent.graph import agent_graph
from src.agent.state import AgentState
from config.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

judge_model = ChatGroq(
    model="llama-3.1-8b-instant",   # cheaper, faster
    api_key=settings.groq_api_key,
    temperature=0.0,
)
ragas_llm = LangchainLLMWrapper(judge_model)

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
)


def load_questions(path: Path) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


async def run_agent_for_question(query: str) -> Dict[str, Any]:
    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "tool_calls_made": [],
        "tool_results": [],
        "final_answer": None,
        "iteration_count": 0,
    }
    config = {"configurable": {"thread_id": f"eval-{hash(query)}"}}

    try:
        final_state = await agent_graph.ainvoke(initial_state, config=config)
        answer = final_state.get("final_answer", "")
    except Exception as e:
        logger.error(f"Agent failed on question '{query}': {e}")
        return {"answer": f"[ERROR: Agent failed to answer this question: {e}]", "contexts": []}

    contexts = []
    for tool_result in final_state.get("tool_results", []):
        tool_name = tool_result["tool_name"]
        result_str = tool_result["result"]

        if tool_name == "search_filings":
            import re
            snippets = re.findall(r"Snippet: (.+?)\.\.\.", result_str)
            contexts.extend(snippets)

        elif tool_name == "sql_query":
            lines = result_str.split("\n")
            for line in lines[2:]:
                if line.strip():
                    contexts.append(line.strip())

        elif tool_name == "financial_ratio_calculator":
            if "Error" not in result_str:
                contexts.append(f"Ratio result: {result_str}")

        elif tool_name == "calculator":
            if "Error" not in result_str:
                contexts.append(f"Calculation result: {result_str}")

    if not contexts:
        contexts = ["[No tool context was retrieved for this question.]"]

    # ✅ FIX: Ensure contexts is ALWAYS a list of strings (defensive validation)
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        contexts = ["[Invalid context format]"]
    
    return {"answer": answer, "contexts": contexts}


async def collect_results(questions_data: List[Dict]) -> List[Dict]:
    results = []
    for item in tqdm(questions_data, desc="Evaluating Agent on Questions"):
        query = item["question"]
        ground_truth = item.get("ground_truth", "")
        output = await run_agent_for_question(query)
        results.append({
            "question": query,
            "answer": output["answer"],
            "contexts": output["contexts"],
            "ground_truth": ground_truth,
        })
    return results


def main():
    dataset_path = Path("evaluation/datasets/eval_questions.json")
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    questions_data = load_questions(dataset_path)
    
    # 🔥 FIX 1: Trim to first 2 questions only to avoid rate limits
    DEBUG_MODE = False
    if DEBUG_MODE:
        logger.info("⚠️ DEBUG MODE: Using only first 2 questions.")
        questions_data = questions_data[:2]   # ✅ Now defined inside main()
    else:
        logger.info("✅ FULL DATASET MODE: Running on all questions.")

    # Run the agent
    results = asyncio.run(collect_results(questions_data))

    # Build RAGAS Dataset
    dataset_dict = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    dataset = Dataset.from_dict(dataset_dict)

    # 🔥 FIX 2: Throttle concurrency and increase timeout
    metrics = [faithfulness]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(max_workers=1, timeout=300),  # sequential, 5 min per question
        raise_exceptions=False,   # don't crash on timeouts
    )
    df = result.to_pandas()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    individual_scores = df.to_dict(orient="records")
    individual_scores_serializable = convert_to_serializable(individual_scores)

    summary = {
        "timestamp": timestamp,
        "metrics": {
            "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else None,
        },
        "individual_scores": individual_scores_serializable,
    }
    
    with open(out_dir / f"ragas_scores_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION SUMMARY (fast iteration mode)")
    print("=" * 50)
    print(f"Faithfulness (mean): {summary['metrics']['faithfulness']:.3f}")
    print(f"Results saved to {out_dir}/ragas_scores_{timestamp}.json")
    print("=" * 50)


if __name__ == "__main__":
    main()
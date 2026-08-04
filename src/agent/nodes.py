import re
import json
import logging
from typing import Any, Dict, List
from groq import AsyncGroq
from config.config import settings
from src.agent.state import AgentState
from src.tools.financial_tools import ( tool_search_filings, tool_sql_query, tool_calculator, tool_financial_ratio_calculator, tool_get_filing_metadata,)

logger = logging.getLogger(__name__)
groq_client = AsyncGroq(api_key=settings.groq_api_key)

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Search JPMorgan Chase SEC filings (10-K, 10-Q, 8-K) using hybrid retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Execute a read-only SQL SELECT query on the financial metrics database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT statement"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a mathematical expression (+, -, *, /, **, %, parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_ratio_calculator",
            "description": "Compute common financial ratios: ROE, Debt_Equity, ROA, Current_Ratio, Quick_Ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ratio_name": {
                        "type": "string",
                        "description": "One of: ROE, Debt_Equity, ROA, Current_Ratio, Quick_Ratio",
                    },
                    "net_income": {"type": "number", "description": "Net income (for ROE, ROA)"},
                    "equity": {"type": "number", "description": "Stockholders' equity (for ROE, Debt_Equity)"},
                    "assets": {"type": "number", "description": "Total assets (for ROA)"},
                    "total_liabilities": {"type": "number", "description": "Total liabilities (for Debt_Equity)"},
                    "current_assets": {"type": "number", "description": "Current assets (for Current_Ratio, Quick_Ratio)"},
                    "current_liabilities": {"type": "number", "description": "Current liabilities (for Current_Ratio, Quick_Ratio)"},
                    "inventory": {"type": "number", "description": "Inventory (for Quick_Ratio, optional, default 0)"},
                },
                "required": ["ratio_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing_metadata",
            "description": "Return a summary of available data: text chunks, filing types, and financial metrics coverage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def extract_numeric_values(sql_result: str) -> set:
    raw = re.findall(r'\d+\.?\d*', sql_result)
    return {float(n) for n in raw if n}


async def reasoning_node(state: AgentState) -> Dict[str, Any]:
    logger.info(f"Reasoning step {state['iteration_count'] + 1}")

    max_iter = settings.max_iterations
    force_final = state["iteration_count"] >= max_iter

    messages = state.get("messages", [])
    if not messages:
        system_prompt = (
            "You are a financial assistant for JPMorgan Chase & Co. "
            "You have access to these tools:\n"
            "- search_filings: hybrid search over SEC filing text (for qualitative/narrative questions)\n"
            "- sql_query: read-only SQL over a table `financial_metrics` with columns "
            "(metric_name, value, unit, period_end, form_type, filed_date). "
            "Available metric_name values: NetIncomeLoss, Revenues, EarningsPerShareDiluted, "
            "EarningsPerShareBasic, Assets, Liabilities, StockholdersEquity, OperatingIncomeLoss. "
            "There is no company_name or ticker column — this table only contains JPMorgan data.\n"
            "- calculator: arithmetic\n"
            "- financial_ratio_calculator: computes ratios (ROE, ROA, Debt_Equity, Current_Ratio, Quick_Ratio) "
            "given numeric inputs\n"
            "- get_filing_metadata: discover available data ranges\n\n"
            "CRITICAL RULES:\n"
            "1. NEVER invent, estimate, or guess numeric values. Always retrieve real figures via sql_query "
            "or search_filings first.\n"
            "2. To compute a ratio, first query sql_query for the exact metric values you need "
            "(e.g., most recent NetIncomeLoss and StockholdersEquity), THEN call financial_ratio_calculator "
            "with those real retrieved numbers.\n"
            "3. If you don't have enough real data yet, call a retrieval tool — do not answer with placeholder numbers.\n"
            "4. Only compute the ratio(s) the user explicitly asked for — do not calculate additional ratios "
            "unless requested.\n"
            "5. Before calling financial_ratio_calculator, you MUST have retrieved EVERY numeric input "
            "(e.g., both net_income AND equity for ROE) via sql_query in this conversation. "
            "Never supply a number you have not just retrieved.\n"
            "6. Once you have successfully computed the ratio the user asked for, STOP and answer immediately. "
            "Do not continue calling tools after you have the answer."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]

    final_messages = messages
    if force_final:
        final_messages = messages + [{
            "role": "system",
            "content": "You can no longer call any tools. Based on everything retrieved so far, "
                       "give your final answer now in plain natural language. Do not attempt to call a function."
        }]

    response = await groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=final_messages,
        tools=None if force_final else GROQ_TOOLS,
        tool_choice="none" if force_final else "auto",
        temperature=0.2,
        parallel_tool_calls=False,
    )

    assistant_message = response.choices[0].message
    clean_msg = {"role": "assistant", "content": assistant_message.content}

    tool_calls = assistant_message.tool_calls
    if tool_calls:
        clean_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]

        tool_calls_to_execute = [
            {"id": tc.id, "name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
            for tc in tool_calls
        ]

        return {
            "messages": [clean_msg],
            "tool_calls_made": [tc["name"] for tc in tool_calls_to_execute],
            "pending_tool_calls": tool_calls_to_execute,
            "iteration_count": state["iteration_count"] + 1,
        }
    else:
        return {
            "messages": [clean_msg],
            "final_answer": assistant_message.content,
            "iteration_count": state["iteration_count"] + 1,
        }


async def tool_execution_node(state: AgentState) -> Dict[str, Any]:
    pending = state.get("pending_tool_calls", [])
    if not pending:
        return {"tool_results": []}

    retrieved_numbers = set()
    for r in state.get("tool_results", []):
        if r["tool_name"] == "sql_query":
            retrieved_numbers.update(extract_numeric_values(r["result"]))

    results = []
    for tc in pending:
        tool_name = tc["name"]
        args = tc["arguments"]

        if tool_name == "financial_ratio_calculator":
            numeric_args = {k: v for k, v in args.items() if isinstance(v, (int, float))}
            unverified = [k for k, v in numeric_args.items() if float(v) not in retrieved_numbers]
            if unverified:
                logger.warning(f"Rejected unverified inputs: {unverified}")
                results.append({
                    "tool_call_id": tc["id"],
                    "tool_name": tool_name,
                    "result": (
                        f"Error: The value(s) for {unverified} were not found in any sql_query "
                        f"result in this conversation. You must call sql_query to retrieve the "
                        f"real value before using it. Do not guess or estimate."
                    ),
                })
                continue

        logger.info(f"Calling tool: {tool_name} with args {args}")
        try:
            if tool_name == "search_filings":
                result = await tool_search_filings(**args)
            elif tool_name == "sql_query":
                result = tool_sql_query(**args)
            elif tool_name == "calculator":
                result = tool_calculator(**args)
            elif tool_name == "financial_ratio_calculator":
                result = tool_financial_ratio_calculator(**args)
            elif tool_name == "get_filing_metadata":
                result = tool_get_filing_metadata()
            else:
                result = f"Error: Unknown tool '{tool_name}'"
            results.append({"tool_call_id": tc["id"], "tool_name": tool_name, "result": result})
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            results.append({"tool_call_id": tc["id"], "tool_name": tool_name, "result": f"Error: {e}"})

    new_tool_messages = [
        {"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["result"]}
        for r in results
    ]

    for r in results:
        if r["tool_name"] == "financial_ratio_calculator" and "Error" not in r["result"]:
            return {
                "messages": new_tool_messages,
                "tool_results": state.get("tool_results", []) + results,
                "pending_tool_calls": [],
                "iteration_count": 999,
            }

    return {
        "messages": new_tool_messages,
        "tool_results": state.get("tool_results", []) + results,
        "pending_tool_calls": [],
    }

async def finalizer_node(state: AgentState) -> Dict[str, Any]:
    if state.get("final_answer"):
        return {"final_answer": state["final_answer"]}

    messages = state["messages"]
    system_message = {
        "role": "system",
        "content": (
            "You are a financial assistant. Based on the conversation history (including tool results), "
            "provide a clear, concise, and accurate answer to the user's original query. "
            "If you don't have enough information, say so. "
            "Do not mention internal tool names; just give the final answer. "
            "Do not use LaTeX or markdown formatting like \\boxed{}; just write plain text."
        )
    }
    full_messages = [system_message] + messages

    response = await groq_client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=full_messages,
        temperature=0.3,
    )

    final_answer = response.choices[0].message.content
    final_answer = re.sub(r'\$\\boxed\{([^}]+)\}\$', r'\1', final_answer)
    final_answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', final_answer)
    return {"final_answer": final_answer}
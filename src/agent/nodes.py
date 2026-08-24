import re
import json
import logging
from typing import Any, Dict, List
from groq import AsyncGroq
from config.config import settings
from src.agent.state import AgentState
from src.tools.financial_tools import (
    tool_search_filings,
    tool_sql_query,
    tool_calculator,
    tool_financial_ratio_calculator,
    tool_get_filing_metadata,
)
from langfuse.decorators import observe
from langfuse.decorators import langfuse_context

logger = logging.getLogger(__name__)
groq_client = AsyncGroq(api_key=settings.groq_api_key)

# ------------------------------------------------------------------
# Global tool definitions – used by the router filter
# ------------------------------------------------------------------
GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_filings",
            "description": "Hybrid search over SEC filing text for qualitative/narrative questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Execute a read‑only SQL SELECT query on the financial_metrics table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT statement"}
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a mathematical expression (+, -, *, /, **, %, parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g., '2 + 2'"}
                },
                "required": ["expression"]
            }
        }
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
                        "enum": ["ROE", "Debt_Equity", "ROA", "Current_Ratio", "Quick_Ratio"],
                        "description": "The ratio to compute"
                    },
                    "net_income": {
                        "type": "number",
                        "description": "Net income – retrieve via sql_query with metric_name = 'NetIncomeLoss' (case-sensitive). Required for ROE, ROA."
                    },
                    "equity": {
                        "type": "number",
                        "description": "Stockholders' equity – retrieve via sql_query with metric_name = 'StockholdersEquity' (case-sensitive). Required for ROE, Debt_Equity."
                    },
                    "assets": {
                        "type": "number",
                        "description": "Total assets – retrieve via sql_query with metric_name = 'Assets' (case-sensitive). Required for ROA."
                    },
                    "total_liabilities": {
                        "type": "number",
                        "description": "Total liabilities – retrieve via sql_query with metric_name = 'Liabilities' (case-sensitive). Required for Debt_Equity."
                    },
                    "current_assets": {
                        "type": "number",
                        "description": "Current assets – retrieve via sql_query with metric_name = 'Assets' (case-sensitive). Required for Current_Ratio, Quick_Ratio."
                    },
                    "current_liabilities": {
                        "type": "number",
                        "description": "Current liabilities – retrieve via sql_query with metric_name = 'Liabilities' (case-sensitive). Required for Current_Ratio, Quick_Ratio."
                    },
                    "inventory": {
                        "type": "number",
                        "description": "Inventory (optional, default 0). Used for Quick_Ratio."
                    }
                },
                "required": ["ratio_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_filing_metadata",
            "description": "Return a summary of available data: text chunks, filing types, and financial metrics coverage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ------------------------------------------------------------------
# 1. INTENT ROUTER (Bouncer) – Decides which tools are allowed
# ------------------------------------------------------------------
def route_intent(query: str) -> List[str]:
    q = query.lower()
    # Ratio questions
    if any(word in q for word in ["roe", "roa", "debt_equity", "current_ratio", "quick_ratio", "ratio"]):
        return ["sql_query", "financial_ratio_calculator"]
    
    # 🔥 FIXED: Single metric – ONLY sql_query, NO calculator
    if any(word in q for word in ["net income", "assets", "liabilities", "equity", "revenue", "earnings", "total assets"]):
        return ["sql_query"]  # <-- Removed 'calculator'
    
    # Narrative / qualitative
    if any(word in q for word in ["risk", "factors", "business", "overview", "describe", "explain", "challenges"]):
        return ["search_filings", "get_filing_metadata"]
    
    return ["search_filings", "sql_query", "calculator", "financial_ratio_calculator", "get_filing_metadata"]

# ------------------------------------------------------------------
# 2. POST-FILTER GUARDRAIL (Check numbers before final output)
# ------------------------------------------------------------------
def guardrail_verify_numbers(answer: str, contexts: List[str]) -> bool:
    """Returns False if a number in the answer is NOT found in the contexts."""
    nums_in_answer = re.findall(r'\d+\.?\d*', answer)
    if not nums_in_answer:
        return True  # No numbers to verify
    full_context = " ".join(contexts)
    for num in nums_in_answer:
        if num not in full_context:
            return False
    return True

def extract_numeric_values(sql_result: str) -> set:
    raw = re.findall(r'\d+\.?\d*', sql_result)
    return {float(n) for n in raw if n}


# ------------------------------------------------------------------
# 3. REASONING NODE (Uses the Intent Router)
# ------------------------------------------------------------------
@observe(as_type="generation")
async def reasoning_node(state: AgentState) -> Dict[str, Any]:
    logger.info(f"Reasoning step {state.get('iteration_count', 0) + 1}")

    # ------------------------------------------------------------------
    # 1. Determine if we must force a final answer
    # ------------------------------------------------------------------
    current_iter = state.get("iteration_count", 0)
    force_final = current_iter >= settings.max_iterations

    # ------------------------------------------------------------------
    # 2. Build / retrieve message history
    # ------------------------------------------------------------------
    messages = state.get("messages", [])

    if not messages:
        # Base system prompt (only built on the very first call)
        system_prompt = (
            "You are a financial assistant for JPMorgan Chase & Co.\n\n"
            "⚠️ DATABASE SCHEMA (CASE-SENSITIVE!):\n"
            "Table: financial_metrics\n"
            "Columns: id (ignored), metric_name, value, unit, period_end, form_type, filed_date.\n"
            "Valid metric_name values (case‑sensitive): Assets, EarningsPerShareBasic, EarningsPerShareDiluted, Liabilities, NetIncomeLoss, OperatingIncomeLoss, Revenues, StockholdersEquity.\n"
            "🔴 NO other columns exist (no 'company', 'ticker', 'date', or 'net_income').\n\n"
            "📌 EXACT RULES:\n"
            "1. SINGLE METRIC (e.g., 'net income', 'total assets'):\n"
            "   → Run EXACTLY ONE SQL: SELECT value FROM financial_metrics WHERE metric_name = '<EXACT_NAME>' ORDER BY period_end DESC LIMIT 1;\n"
            "   → STOP immediately. Do NOT run calculator or any other tool.\n"
            "2. RATIO (e.g., 'ROE', 'ROA'):\n"
            "   → Run SQL to get the required metrics, then call financial_ratio_calculator with those exact numbers.\n"
            "3. QUALITATIVE (e.g., 'risk factors'):\n"
            "   → Call search_filings ONCE and STOP.\n"
            "4. NEVER guess numbers. Use JSON function-calling format."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["query"]}
        ]

    # ------------------------------------------------------------------
    # 3. If force_final, add a system message to stop tool calls
    # ------------------------------------------------------------------
    if force_final:
        messages = messages + [{
            "role": "system",
            "content": "You can no longer call tools. Answer now in plain text."
        }]

    # ------------------------------------------------------------------
    # 4. Filter tools based on intent & inject restriction into messages
    # ------------------------------------------------------------------
    # Default values (safe fallback)
    tools_to_use = None
    tool_choice = "none"

    if not force_final:
        allowed_names = route_intent(state["query"])
        tools_to_use = [t for t in GROQ_TOOLS if t["function"]["name"] in allowed_names]
        tool_choice = "auto"

        # 🔥 FIX: Inject the tool restriction directly into the messages list
        # so the LLM actually sees it before making the API call.
        restriction_msg = (
            f"🔒 You are ONLY allowed to use these tools: {', '.join(allowed_names)}. "
            "If a tool is not in this list, do NOT attempt to call it. "
            "NEVER use XML tags like <function=...>. Use the JSON tool-calling interface."
        )

        # Update the existing system message with the restriction
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{restriction_msg}"
        else:
            # Safety: if there's no system message, insert one
            messages.insert(0, {"role": "system", "content": restriction_msg})

    # ------------------------------------------------------------------
    # 5. Call Groq with proper error handling
    # ------------------------------------------------------------------
    assistant_content = None
    tool_calls = []
    api_error = None

    try:
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            tools=tools_to_use,
            tool_choice=tool_choice,
            temperature=0.2,
            parallel_tool_calls=False,
        )
        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""
        tool_calls = assistant_message.tool_calls or []

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        api_error = str(e)

    # ------------------------------------------------------------------
    # 6. Update Langfuse observation (safely)
    # ------------------------------------------------------------------
    try:
        langfuse_context.update_current_observation(
            name="groq_llm_call",
            input={"messages": messages},
            output={"content": assistant_content, "tool_calls": tool_calls},
            metadata={
                "model": settings.GROQ_MODEL,
                "temperature": 0.2,
                "tool_choice": tool_choice,
                "forced_final": force_final,
                "api_error": api_error,
            },
        )
    except Exception as e:
        logger.warning(f"Langfuse update failed: {e}")

    # ------------------------------------------------------------------
    # 7. Build return state
    # ------------------------------------------------------------------
    if assistant_content is None:
        assistant_content = "I encountered an internal error while processing your request."

    clean_msg = {"role": "assistant", "content": assistant_content}

    # If API error occurred, treat it as a final answer
    if api_error:
        return {
            "messages": [clean_msg],
            "final_answer": assistant_content,
            "iteration_count": current_iter + 1,
        }

    # If there are tool calls, parse and return them
    if tool_calls:
        parsed_tool_calls = []
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            parsed_tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": args,
            })

        clean_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in tool_calls
        ]

        return {
            "messages": [clean_msg],
            "tool_calls_made": [tc["name"] for tc in parsed_tool_calls],
            "pending_tool_calls": parsed_tool_calls,
            "iteration_count": current_iter + 1,
        }

    # No tool calls – final answer
    return {
        "messages": [clean_msg],
        "final_answer": assistant_content,
        "iteration_count": current_iter + 1,
    }
# ------------------------------------------------------------------
# 4. TOOL EXECUTION NODE
# ------------------------------------------------------------------
@observe(as_type="span")
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
                error_msg = (
                    f"Error: The value(s) for {unverified} were not found in any sql_query "
                    f"result. You must call sql_query to retrieve the real value before using it."
                )
                results.append({
                    "tool_call_id": tc["id"],
                    "tool_name": tool_name,
                    "result": error_msg,
                })
                langfuse_context.update_current_observation(level="WARNING")
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
            
            if "Error" in result:
                langfuse_context.update_current_observation(level="ERROR")

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}")
            error_result = f"Error: {e}"
            results.append({"tool_call_id": tc["id"], "tool_name": tool_name, "result": error_result})
            langfuse_context.update_current_observation(level="ERROR")

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


# ------------------------------------------------------------------
# 5. FINALIZER NODE (with Post-Filter Guardrail)
# ------------------------------------------------------------------
@observe(as_type="span")
async def finalizer_node(state: AgentState) -> Dict[str, Any]:
    # Early exit if answer already exists
    if state.get("final_answer"):
        return {"final_answer": state["final_answer"]}

    # Default fallback
    final_answer = "I couldn't generate a proper answer."

    messages = state["messages"]
    system_message = {
        "role": "system",
        "content": (
            "Provide a clear, concise answer to the user's original query using the tool results. "
            "If the user asked for a number, just state the number. Do not mention internal tools."
        )
    }
    full_messages = [system_message] + messages

    try:
        response = await groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=full_messages,
            temperature=0.3,
        )
        if response.choices and response.choices[0].message.content is not None:
            final_answer = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Finalizer API call failed: {e}")

    # Clean up markdown if present
    try:
        final_answer = re.sub(r'\$\\boxed\{([^}]+)\}\$', r'\1', final_answer)
        final_answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', final_answer)
    except Exception:
        pass

    # 🛡️ POST-FILTER GUARDRAIL
    all_contexts = []
    for tr in state.get("tool_results", []):
        all_contexts.append(tr["result"])

    if not guardrail_verify_numbers(final_answer, all_contexts):
        logger.warning("🔴 Guardrail triggered: Unverified numbers found. Blocking hallucination.")
        final_answer = "I couldn't verify that exact number in the retrieved documents. Please check the source."

    try:
        langfuse_context.update_current_span(
            name="groq_finalizer",
            input={"messages": full_messages},
            output={"content": final_answer},
            metadata={"model": settings.GROQ_MODEL, "temperature": 0.3},
        )
    except Exception:
        pass

    return {"final_answer": final_answer}
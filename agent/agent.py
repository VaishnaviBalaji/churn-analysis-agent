"""
Core agent loop using the Claude API with tool use.
The agent receives a natural language question and autonomously
decides which tools to call, interprets results, and returns a report.
"""

import os
import json
import pandas as pd
import anthropic
from agent.tools import (
    TOOL_DEFINITIONS,
    score_customers,
    analyze_segment,
    get_high_risk_customers,
    bucket_distribution,
)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are an autonomous data analyst specialising in customer churn.
You have access to a set of tools to analyse a customer dataset.

Your job:
1. Understand the user's question.
2. Call the relevant tools — you may call multiple tools in sequence.
3. Interpret the results and write a clear, concise analysis report.

Guidelines:
- Always start by scoring customers if scores are not yet available.
- Be specific: quote numbers, percentages, and segment names.
- End with 1-3 concrete retention recommendations based on what you found.
- Keep the tone professional but plain — this will be read by a business stakeholder.
"""


def run_agent(question: str, df: pd.DataFrame) -> tuple[str, list[dict]]:
    """
    Run the agent loop for a given question and customer dataframe.

    Returns:
        - final_report (str): The agent's natural language analysis
        - tool_calls_log (list[dict]): Log of every tool called and its result
    """
    # We hold df in this closure so tools can access it without passing via API
    scored_df = [df.copy()]  # mutable container

    messages = [{"role": "user", "content": question}]
    tool_calls_log = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Collect any text content so far
        text_blocks = [b.text for b in response.content if b.type == "text"]

        # If Claude is done (no more tool calls), return the final text
        if response.stop_reason == "end_turn":
            return "\n".join(text_blocks), tool_calls_log

        # Process tool use blocks
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            result = _dispatch_tool(tool_name, tool_input, scored_df)

            tool_calls_log.append({
                "tool": tool_name,
                "input": tool_input,
                "result": result
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result)
            })

        # Feed tool results back into the conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def _dispatch_tool(tool_name: str, tool_input: dict, scored_df: list) -> dict:
    """Route a tool call to the correct implementation."""
    df = scored_df[0]

    if tool_name == "score_customers":
        scored_df[0] = score_customers(df)
        return {"status": "scored", "row_count": len(scored_df[0])}

    elif tool_name == "analyze_segment":
        return analyze_segment(scored_df[0], tool_input["segment_col"])

    elif tool_name == "get_high_risk_customers":
        return get_high_risk_customers(scored_df[0], tool_input.get("bucket", "critical"))

    elif tool_name == "bucket_distribution":
        return bucket_distribution(scored_df[0])

    return {"error": f"Unknown tool: {tool_name}"}

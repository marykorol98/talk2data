# prompts.py

import json
from typing import TypedDict

from string import Template


class PromptMessage(TypedDict):
    role: str
    content: str


# Decision action prompt as chat messages
DECIDE_ACTION_PROMPT: list[PromptMessage] = [
    {
        "role": "system",
        "content": (
            "You are a classification assistant that must decide the correct action type for a given user request.\n\n"
            "You always choose one of two actions:\n"
            "1. code_generation — when the user is asking to create, modify, or execute code, especially for data analysis, visualization, or manipulation.\n"
            "2. chat_response — when the user is asking a general question, explanation, or non-technical request.\n\n"
            "### Decision rules:\n"
            "- Choose **code_generation** if the request mentions or implies any of the following:\n"
            "  words such as *plot, draw, show chart, graph, visualize, histogram, scatter, table, dataframe, generate code, python, calculate, compute, analyze, correlation, regression, model, column, dataset, data analysis*.\n"
            "- Choose **code_generation** if the user asks for a visualization, computation, or any action that would normally require programming or code.\n"
            "- Choose **chat_response** for conversational, descriptive, or explanatory requests (e.g., 'explain this concept', 'what does this mean?', 'what columns exist?').\n"
            "- When uncertain, prefer **code_generation** if the user’s message contains technical language or data terms.\n\n"
            "### Input context:\n"
            "Dataset metadata: $metadata\n\n"
            "### Output format:\n"
            "Respond ONLY with a valid JSON object on a single line in the exact format:\n"
            '{{"action": "code_generation"}} or {{"action": "chat_response"}}\n\n'
            "### Examples:\n"
            '- \'Show distribution of sales\' -> {"action": "code_generation"}\n'
            '- \'Plot a histogram for the data\' -> {"action": "code_generation"}\n'
            '- \'Visualize relationship between age and income\' -> {"action": "code_generation"}\n'
            '- \'What columns are available?\' -> {"action": "chat_response"}\n'
            '- \'Explain this code\' -> {"action": "chat_response"}\n'
            '- \'What is a histogram?\' -> {"action": "chat_response"}\n'
            "\n"
            "Return nothing else — no explanation, no notes, just the JSON."
        ),
    },
    {
        "role": "user",
        "content": (
            "Conversation History:\n$history\n\n"
            "User Request:\n$input\n\n"
            "Your output must be ONLY one valid JSON object with the key 'action'."
        ),
    },
]


# Chat response prompt as chat messages
CHAT_RESPONSE_PROMPT: list[PromptMessage] = [
    {
        "role": "system",
        "content": (
            "You are a friendly assistant helping the user understand data.\n"
            "Do not write code.\n"
            "Respond in simple, clear sentences suitable for reading aloud by a Text-to-Speech (TTS) system.\n"
            "Use **Markdown** for formatting. It should be minimal and readable aloud.\n"
            "Allowed Markdown:\n"
            "- Paragraphs\n"
            "- Simple bullet lists\n"
            "- Bold for key terms\n"
            "- Code blocks for examples\n"
            "Do not use tables, HTML, or complex formatting.\n\n"
            "Always use natural, spoken language.\n\n"
            "Current dataset details:\n$metadata\n\n"
            "Conversation history:\n$history\n\n"
            "If you're unsure about the user's request, ask for clarification.\n"
            "If the question is technical or requires code, politely suggest generating Python code instead.\n"
            "Be brief and precise.\n"
        ),
    },
    {"role": "user", "content": "Question: $input"},
]


# Code generation prompt as chat messages
CODE_GENERATION_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are a data science expert. Generate Python code only for DataFrame 'df' with only these columns:\n"
            "$metadata\n\n"
            "Here is the conversation History:\n$history\n"
            "Instructions:\n"
            "1. Use Plotly. You must not use pyplot or seaborn, only Plotly.\n"
            "2. Work strictly with the provided DataFrame 'df' — do not read or write any external files, do not call URLs, and do not construct new data.\n"
            "3. When a user requests basic statistics or a quick overview, return concise Pandas operations like df.describe() instead of plots. Create visualizations only when explicitly requested.\n"
            "4. For showing output, use expression form (variable name), not print/display.\n"
            "5. Critical! generate only code without any comments or explanations, just python code!"
        ),
    },
    {"role": "user", "content": "Request: $input"},
]

CODE_BEGIN = "PY_CODE_BEGIN"
CODE_END = "PY_CODE_END"
PLAN_BEGIN = "PLAN_JSON_BEGIN"
PLAN_END = "PLAN_JSON_END"

def _mapping_from_state(state: dict) -> dict:
    return {
        "input": str(state.get("user_input", "")),
        "history": str(state.get("conversation_history", "")),
        "metadata": str(state.get("metadata", {})),
    }
    

def _render_messages(state: dict, extra_user_content: str | None = None) -> list[dict]:
    m = _mapping_from_state(state)
    msgs = [
        {
            "role": "system",
            "content": Template(CODE_GENERATION_PROMPT[0]["content"]).safe_substitute(m),
        },
        {
            "role": "user",
            "content": Template(CODE_GENERATION_PROMPT[1]["content"]).safe_substitute(m),
        },
    ]
    if extra_user_content:
        msgs.append({"role": "user", "content": extra_user_content})
    return msgs

def build_code_prompt(state: dict) -> list[dict]:
    plan_blob = json.dumps(state.get("plan", {}), ensure_ascii=False)

    code_user = (
        "Use the internal JSON plan below to generate Python code.\n"
        "Do NOT output the plan. Do NOT output explanations. Do NOT output comments.\n"
        f"Return ONLY python code wrapped strictly between markers:\n"
        f"{CODE_BEGIN}\n"
        "<python code>\n"
        f"{CODE_END}\n\n"
        f"Internal plan JSON:\n{plan_blob}\n\n"
        "Hard rules:\n"
        "- Only use df and its existing columns. DO NOT use pd.read_csv, df is already in the context, use this variable as it is\n"
        "- No external files, URLs, or fabricated data.\n"
        "- When a user requests basic statistics or a quick overview, return concise Pandas operations instead of plots. Create visualizations only when explicitly requested.\n"
        "- For plots use plotly only (px or go).\n"
        "- No markdown fences.\n"
    )

    messages = _render_messages(state, extra_user_content=code_user)
    return messages


def build_plan_prompt(state: dict) -> list[dict]:
    plan_user = (
        "Produce a SHORT internal execution plan as JSON.\n"
        "Do NOT provide reasoning. Do NOT provide any text outside the JSON.\n"
        f"Return ONLY between markers {PLAN_BEGIN} and {PLAN_END}.\n\n"
        f"{PLAN_BEGIN}\n"
        "{\n"
        '  "intent": "stats" | "plot",\n'
        '  "columns_used": ["col1", "col2"],\n'
        '  "ops": ["step1", "step2"],\n'
        '  "plot_spec": {"kind": "line|bar|scatter|hist|box|heatmap|other", "x": "...", "y": "...", "color": null, "facet": null},\n'
        '  "notes": "short notes (e.g., convert to datetime) or empty"\n'
        "}\n"
        f"{PLAN_END}\n\n"
        "Rules:\n"
        "- If the user did NOT explicitly ask for a plot/visualization, set intent='stats'.\n"
        "- Use only columns that exist in the provided metadata.\n"
    )
    messages = _render_messages(state, extra_user_content=plan_user)
    return messages
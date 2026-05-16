"""
agent.py — Conversational SHL Assessment Recommender
using OpenRouter + Retrieval

Features:
- OpenRouter LLM integration
- Retrieval grounded responses
- Hallucination protection
- JSON-safe output parsing
- Off-topic protection
- Comparison support
"""

import json
import os
import re
from typing import Any

from openai import OpenAI

import retriever


# -------------------------------------------------------------------
# OpenRouter Setup
# -------------------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY environment variable not set."
    )

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"


# -------------------------------------------------------------------
# System Prompt
# -------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an SHL Assessment Recommender.

You help recruiters and hiring managers find the
right SHL assessments.

STRICT RULES:
1. ONLY recommend assessments from the provided catalog.
2. NEVER invent names or URLs.
3. ONLY use URLs exactly as provided.
4. Stay strictly on SHL assessments.
5. If the query is vague, ask ONE clarification question.
6. Max 10 recommendations.
7. Use ONLY JSON output.

OUTPUT FORMAT:

{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "string",
      "test_type": "string"
    }
  ],
  "end_of_conversation": false
}
"""


# -------------------------------------------------------------------
# JSON Extraction
# -------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Safely extract JSON from model output.
    """

    if not text:
        return {
            "reply": "Empty model response.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Fallback
    return {
        "reply": text,
        "recommendations": [],
        "end_of_conversation": False,
    }


# -------------------------------------------------------------------
# Retrieval Query Builder
# -------------------------------------------------------------------

def _build_retrieval_query(
    messages: list[dict]
) -> str:

    user_messages = [
        m["content"]
        for m in messages
        if m["role"] == "user"
    ]

    return " ".join(user_messages[-3:])


# -------------------------------------------------------------------
# Catalog Context Builder
# -------------------------------------------------------------------

def _build_catalog_context(
    messages: list[dict]
) -> str:

    query = _build_retrieval_query(messages)

    top_items = retriever.search(
        query,
        k=15
    )

    lines = [
        "CATALOG ITEMS:"
    ]

    for item in top_items:

        name = item.get("name", "")
        link = item.get("link", "")
        desc = item.get("description", "")
        job_levels = ", ".join(
            item.get("job_levels", [])
        )

        keys = ", ".join(
            item.get("keys", [])
        )

        lines.append(
            f"""
NAME: {name}
URL: {link}
JOB_LEVELS: {job_levels}
KEYS: {keys}
DESCRIPTION: {desc[:300]}
"""
        )

    return "\n".join(lines)


# -------------------------------------------------------------------
# Off-topic Protection
# -------------------------------------------------------------------

def _is_off_topic(message: str) -> bool:

    patterns = [
        r"ignore previous",
        r"act as",
        r"salary",
        r"lawsuit",
        r"legal advice",
        r"competitor",
    ]

    msg = message.lower()

    return any(
        re.search(p, msg)
        for p in patterns
    )


# -------------------------------------------------------------------
# Main Chat Function
# -------------------------------------------------------------------

def chat(
    messages: list[dict]
) -> dict[str, Any]:

    # ---------------------------------------------------------------
    # Empty conversation
    # ---------------------------------------------------------------

    if not messages:

        return {
            "reply": (
                "Hello! Tell me about the role "
                "you're hiring for."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ---------------------------------------------------------------
    # Last user message
    # ---------------------------------------------------------------

    last_user_msg = next(
        (
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user"
        ),
        "",
    )

    # ---------------------------------------------------------------
    # Off-topic guard
    # ---------------------------------------------------------------

    if _is_off_topic(last_user_msg):

        return {
            "reply": (
                "I can only help with SHL "
                "assessment recommendations."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ---------------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------------

    catalog_context = _build_catalog_context(
        messages
    )

    # ---------------------------------------------------------------
    # Build conversation text
    # ---------------------------------------------------------------

    conversation_text = ""

    for msg in messages:

        role = msg["role"].upper()

        conversation_text += (
            f"{role}: "
            f"{msg['content']}\n"
        )

    # ---------------------------------------------------------------
    # Final prompt
    # ---------------------------------------------------------------

    prompt = f"""
{catalog_context}

CONVERSATION:

{conversation_text}

Remember:
- Only recommend from catalog
- Use exact URLs
- Return ONLY valid JSON
"""

    # ---------------------------------------------------------------
    # LLM Call
    # ---------------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )

        raw_text = (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return {
            "reply": (
                f"LLM error: {str(e)}"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # ---------------------------------------------------------------
    # Parse JSON
    # ---------------------------------------------------------------

    result = _extract_json(raw_text)

    # ---------------------------------------------------------------
    # Validate recommendations
    # ---------------------------------------------------------------

    catalog = retriever.get_all()

    valid_urls = {
        item.get("link", "")
        for item in catalog
    }

    cleaned = []

    for rec in result.get(
        "recommendations",
        []
    ):

        if (
            rec.get("url")
            in valid_urls
        ):

            cleaned.append({
                "name": rec.get(
                    "name",
                    ""
                ),
                "url": rec.get(
                    "url",
                    ""
                ),
                "test_type": rec.get(
                    "test_type",
                    ""
                ),
            })

    # Limit to 10
    cleaned = cleaned[:10]

    # ---------------------------------------------------------------
    # Final safe response
    # ---------------------------------------------------------------

    return {
        "reply": result.get(
            "reply",
            ""
        ),
        "recommendations": cleaned,
        "end_of_conversation": bool(
            result.get(
                "end_of_conversation",
                False
            )
        ),
    }
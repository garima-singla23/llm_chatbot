"""Agentic observe-plan-act loop for the airline assistant."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .tool_definitions import TOOL_REGISTRY


load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


AGENT_SYSTEM_PROMPT = """You are an airline assistant with tools.
Use tools for: flight status, booking lookup, refund calculation, baggage tracking.
For policy questions, answer directly without tools.
Always use valid JSON when calling functions.
Be brief and helpful."""


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
                "description": "Search flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                            "description": "Origin city or code",
                    },
                    "destination": {
                        "type": "string",
                            "description": "Destination city or code",
                    },
                    "date": {
                        "type": "string",
                            "description": "Travel date",
                    },
                },
                "required": ["origin", "destination"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_flight_status",
                "description": "Check a flight status",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_number": {
                        "type": "string",
                            "description": "Flight number",
                    }
                },
                "required": ["flight_number"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking_page_url",
                "description": "Create a booking page link",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight": {
                        "type": "object",
                            "description": "Selected flight object",
                    },
                    "origin": {
                        "type": "string",
                            "description": "Origin city or code",
                    },
                    "destination": {
                        "type": "string",
                            "description": "Destination city or code",
                    },
                    "date": {
                        "type": "string",
                            "description": "Travel date",
                        "default": "",
                    },
                },
                "required": ["flight", "origin", "destination"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_booking_by_pnr",
                "description": "Look up a booking by PNR",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    }
                },
                "required": ["pnr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_booking",
                "description": "Look up or create a booking",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    }
                },
                "required": ["pnr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alternative_flights",
                "description": "Find alternative flights",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                            "description": "Origin city or code",
                    },
                    "destination": {
                        "type": "string",
                            "description": "Destination city or code",
                    },
                    "date": {
                        "type": "string",
                            "description": "Travel date",
                    },
                    "n": {
                        "type": "integer",
                            "description": "Number of flights",
                        "default": 3,
                    },
                },
                "required": ["origin", "destination", "date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rebook_flight",
                "description": "Rebook a flight",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    },
                    "new_flight_no": {
                        "type": "string",
                            "description": "New flight number",
                    },
                },
                "required": ["pnr", "new_flight_no"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
                "description": "Calculate a refund",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    }
                },
                "required": ["pnr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_baggage",
                "description": "Track baggage status",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    }
                },
                "required": ["pnr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_baggage_claim",
                "description": "Create a baggage claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnr": {
                        "type": "string",
                            "description": "Booking reference",
                    },
                    "description": {
                        "type": "string",
                            "description": "Issue description",
                    },
                },
                "required": ["pnr", "description"],
                "additionalProperties": False,
            },
        },
    },
]


def extract_entities(message: str) -> Dict[str, Optional[Any]]:
    """Extract flight numbers, PNRs, and common city names from a message."""
    flight_match = re.search(r"\b[A-Z]{2}\d{3,4}\b", message, flags=re.IGNORECASE)
    pnr_match = re.search(r"\bPNR[A-Z0-9]{6,8}\b", message, flags=re.IGNORECASE)

    city_names = [
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Chennai",
        "Kolkata",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
    ]
    city_found = None
    for city in city_names:
        if re.search(rf"\b{re.escape(city)}\b", message, flags=re.IGNORECASE):
            city_found = city
            break

    return {
        "flight_no": flight_match.group(0).upper() if flight_match else None,
        "pnr": pnr_match.group(0).upper() if pnr_match else None,
        "city": city_found,
    }


def _trim_memory_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return at most the last three exchanges from memory messages."""
    if len(messages) <= 6:
        return messages
    return messages[-6:]


def run_agent(user_message: str, memory, retriever, max_iterations: int = 5) -> str:
    """Run the observe-plan-act agent loop for one user message."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    messages.extend(_trim_memory_messages(memory.to_messages()[-6:]))
    messages.append({"role": "user", "content": user_message})

    tools_called: List[str] = []
    # Cross-iteration cache: tool_key -> result
    # Prevents the LLM from re-invoking the same tool with the same args
    # across multiple iterations of the observe-plan-act loop.
    executed_tools: Dict[str, Any] = {}

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            max_tokens=1000,
            temperature=0.3,
        )

        try:
            tool_call_message = response.choices[0].message

            if response.choices[0].finish_reason == "tool_calls":
                for tool_call in tool_call_message.tool_calls:
                    name = tool_call.function.name

                    # Safely parse arguments
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        print(f"[AGENT] Bad args JSON for {name}: {tool_call.function.arguments}")
                        pnr_match = re.search(r'"pnr":\s*"([^"]+)"', tool_call.function.arguments)
                        flight_match = re.search(r'"flight_number":\s*"([^"]+)"', tool_call.function.arguments)
                        args = {}
                        if pnr_match:
                            args["pnr"] = pnr_match.group(1)
                        if flight_match:
                            args["flight_number"] = flight_match.group(1)
                        if not args:
                            print(f"[AGENT] Could not parse args, skipping tool call")
                            continue

                    if name not in TOOL_REGISTRY:
                        print(f"[AGENT] Unknown tool: {name}")
                        continue

                    tool_key = f"{name}:{json.dumps(args, sort_keys=True)}"

                    # Step 1: Append assistant message declaring the tool call
                    # BEFORE executing the tool (required by OpenAI/Groq format).
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    })

                    # Step 2: Execute the tool — OR reuse cached result if this
                    # exact tool+args was already executed in a prior iteration.
                    if tool_key in executed_tools:
                        print(f"[AGENT] Reusing cached result for duplicate call: {tool_key}")
                        result = executed_tools[tool_key]
                    else:
                        print(f"[AGENT TOOL] {name}({args})")
                        result = TOOL_REGISTRY[name](**args)
                        executed_tools[tool_key] = result
                        tools_called.append(name)

                    # Step 3: Append tool result AFTER the assistant message.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    })

                # After processing all tool_calls in this response, nudge the
                # model to produce a final answer instead of re-calling tools.
                messages.append({
                    "role": "system",
                    "content": (
                        "You have received tool results above. "
                        "Do NOT call the same tool again with the same arguments. "
                        "Produce a final natural-language answer for the user now."
                    ),
                })

                continue

        except Exception as e:
            print(f"[AGENT] Tool execution error: {e}")
            memory.add("user", user_message)
            fallback = (
                "I encountered an issue checking that. "
                "Please try rephrasing as 'Look up booking PNRK1GSE7' "
                "or 'Refund for PNR PNRK1GSE7'."
            )
            memory.add("assistant", fallback)
            return fallback

        choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "stop":
            answer = message.content or ""
            memory.add("user", user_message)
            memory.add("assistant", answer)
            memory.last_tools_used = tools_called
            return answer

        answer = message.content or "I'm sorry, I couldn't complete that request right now."
        memory.add("user", user_message)
        memory.add("assistant", answer)
        memory.last_tools_used = tools_called
        return answer

    return "I've reached the limit of what I can do automatically. Let me connect you with a human agent."
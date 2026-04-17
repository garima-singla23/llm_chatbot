"""
Context Builder for Flight Chatbot LLM.

Merges data from multiple sources (RAG, APIs, intent, entities) into a single
coherent context string for the LLM to reason over.
"""

from typing import Optional, List, Dict, Any

from utils.flightapi_parser import FlightOffer
from utils.flight_formatter import format_for_llm, format_status_for_llm


def filter_citation_sources(sources: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Keep only non-expanded sources for citation display."""
    if not sources:
        return []

    return [source for source in sources if not bool(source.get("expanded", False))]


def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses rough heuristic: 1 token ≈ 4 characters

    Args:
        text: Text to estimate

    Returns:
        Approximate token count
    """
    return len(text) // 4


def section_present(context: str, section: str) -> bool:
    """
    Check if a named section exists in the built context.

    Args:
        context: Built context string
        section: Section name to search for (e.g., "Flight status", "User intent")

    Returns:
        True if section header is found in context
    """
    section_header = f"## {section}"
    return section_header in context


def build_context(
    intent_type: str,
    rag_chunks: Optional[str] = None,
    flight_offers: Optional[List[FlightOffer]] = None,
    flight_status: Optional[Dict[str, Any]] = None,
    entities: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a comprehensive LLM context from multiple data sources.

    Combines intent, search parameters, RAG policy information, flight offers,
    and flight status into a structured context string. Handles token budget
    constraints by truncating less critical sections.

    Args:
        intent_type: User intent classification (e.g., "search_flights", "check_status")
        rag_chunks: Optional retrieved policy/FAQ text from RAG
        flight_offers: Optional list of FlightOffer objects from API
        flight_status: Optional flight status dictionary from API
        entities: Dictionary of extracted entities (origin, destination, date, cabin, adults)

    Returns:
        Formatted context string suitable for LLM consumption, within token limits
    """
    if entities is None:
        entities = {}

    sections = []

    # ============================================================
    # Section A: User Intent and Search Parameters (always present)
    # ============================================================
    origin = entities.get("origin", "Unknown")
    destination = entities.get("destination", "Unknown")
    date = entities.get("date", "Unknown")
    cabin = entities.get("cabin", "ECONOMY")
    adults = entities.get("adults", 1)

    section_a = f"""## User intent
{intent_type}

## Search parameters
Route: {origin} to {destination}
Date: {date}
Cabin: {cabin}
Passengers: {adults} adult(s)"""

    sections.append(section_a)

    # ============================================================
    # Section B: RAG Policy Information (if available and non-empty)
    # ============================================================
    if rag_chunks and rag_chunks.strip():
        section_b = f"## Relevant policy information\n{rag_chunks}"
        sections.append(section_b)

    # ============================================================
    # Section C: Live Flight Results (if available)
    # ============================================================
    if flight_offers is not None:
        query_context = f"{origin} to {destination} on {date}"
        formatted_offers = format_for_llm(flight_offers, query_context)
        section_c = f"## Live flight results\n{formatted_offers}"
        sections.append(section_c)

    # ============================================================
    # Section D: Flight Status (if available)
    # ============================================================
    if flight_status is not None:
        formatted_status = format_status_for_llm(flight_status)
        section_d = f"## Flight status\n{formatted_status}"
        sections.append(section_d)

    # ============================================================
    # Section E: Instructions (always present)
    # ============================================================
    section_e = """## Instructions
Answer using ONLY the information above.
Never invent prices, routes, or policies not shown.
    Cite your source for each claim (policy doc or FlightAPI data).
Be concise and structured."""

    sections.append(section_e)

    # Join sections
    context = "\n\n".join(sections)

    # ============================================================
    # Token Budget Management
    # ============================================================
    max_tokens = 3000
    current_tokens = _estimate_tokens(context)

    if current_tokens > max_tokens:
        # Context exceeds budget, apply truncation strategy
        # Strategy: truncate RAG chunks first, then reduce flight offers

        # Find and truncate RAG chunks section
        if "## Relevant policy information" in context:
            # Find the section
            rag_start = context.find("## Relevant policy information")
            rag_end = context.find("\n##", rag_start + 1)

            if rag_end == -1:
                rag_end = len(context)

            original_rag = context[rag_start:rag_end]

            # Keep only last 1500 chars of RAG chunks
            truncated_rag = original_rag[: len("## Relevant policy information\n")] + original_rag[-1500:]
            context = context[:rag_start] + truncated_rag + context[rag_end:]

            current_tokens = _estimate_tokens(context)

        # If still over budget, truncate flight results to top 3 offers
        if current_tokens > max_tokens and "## Live flight results" in context:
            offers_start = context.find("## Live flight results")
            offers_end = context.find("\n##", offers_start + 1)

            if offers_end == -1:
                offers_end = len(context)

            # Rebuild flight section with only top 3 offers
            if flight_offers and len(flight_offers) > 3:
                top_3 = flight_offers[:3]
                query_context = f"{origin} to {destination} on {date}"
                formatted_top_3 = format_for_llm(top_3, query_context)
                new_offers_section = f"## Live flight results\n{formatted_top_3}"

                context = context[:offers_start] + new_offers_section + context[offers_end:]

    return context

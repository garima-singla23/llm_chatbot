import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from utils.intent import classify_intent
from utils.entity_extractor import extract_entities
from utils.clarifier import needs_clarification, merge_entities
from utils.formatter import build_messages
from apis.aviationstack_client import AviationstackClient
from apis.flightapi_client import FlightAPIClient
from utils.flightapi_parser import parse_flightapi_response
from utils.price_normalizer import normalize_currency, rank_offers, tag_offers
from utils.flight_formatter import format_as_markdown_table
from utils.citation_formatter import format_inline_citations
from utils.confidence_gate import should_fallback, build_fallback_response
from utils.hyde import hyde_retrieve
from pipeline.context_builder import build_context

logger = logging.getLogger(__name__)


def route(
    query: str,
    llm,
    retriever,
    flight_client: FlightAPIClient,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    pending_entities: Optional[Dict] = None,
    aviationstack_client: Optional[AviationstackClient] = None,
) -> Tuple[str, Dict]:
    """Route user query to appropriate handler based on intent classification.

    Args:
        query: User's input query.
        llm: LLM instance with chat(messages, stream=False) method.
        retriever: Retriever instance with retrieve(query) -> str method.
        flight_client: FlightAPI client with search_flights(origin, destination, date, adults, cabin, currency) method.
        chat_history: List of (user, assistant) tuples for context.
        pending_entities: Entities from previous clarification turn (if any).
        aviationstack_client: Optional Aviationstack client for flight status queries.

    Returns:
        (response_text, updated_pending_entities)
        - response_text: str to send to user.
        - updated_pending_entities: dict to persist across turns if clarification is needed.
          Empty dict {} if routing completed (flight search, policy, etc.).
    """
    try:
        if chat_history is None:
            chat_history = []
        if pending_entities is None:
            pending_entities = {}

        # Step 1: Classify intent
        intent_result = classify_intent(query, llm, chat_history)
        intent_type = intent_result.get("type", "policy")
        confidence = intent_result.get("confidence", 0.0)

        # If confidence is too low, default to policy
        if confidence < 0.4:
            intent_type = "policy"

        # Step 2: Route based on intent type
        if intent_type == "flight_search":
            return _handle_flight_search(
                query, llm, flight_client, retriever, chat_history, pending_entities, price_compare=False
            )

        elif intent_type == "price_compare":
            return _handle_flight_search(
                query, llm, flight_client, retriever, chat_history, pending_entities, price_compare=True
            )

        elif intent_type == "policy":
            return _handle_policy(query, llm, retriever)

        elif intent_type == "flight_status":
            return _handle_flight_status(query, llm, aviationstack_client)

        elif intent_type == "general":
            return _handle_general(query, llm, chat_history)

        else:
            # Unknown intent, default to policy
            return _handle_policy(query, llm, retriever)

    except Exception as e:
        logger.error(f"Unhandled error in router: {e}", exc_info=True)
        return (
            "I encountered an unexpected error. Please try again.",
            {},
        )


def _retrieve_docs(retriever, query: str):
    """Support both callable retrievers and retriever objects with retrieve()."""
    if callable(retriever):
        return retriever(query)

    retrieve_fn = getattr(retriever, "retrieve", None)
    if callable(retrieve_fn):
        return retrieve_fn(query)

    raise TypeError("Retriever must be callable or expose a callable retrieve(query) method.")


def _retrieve_context_and_sources(retriever, query: str) -> tuple[str, list[dict]]:
    result = _retrieve_docs(retriever, query)

    if isinstance(result, tuple) and len(result) == 2:
        context, sources = result
        return str(context or ""), list(sources or [])

    return _to_text_chunks(result), []


def _to_text_chunks(chunks) -> str:
    """Convert retriever outputs into a text block for context builder."""
    if chunks is None:
        return ""

    if isinstance(chunks, str):
        return chunks

    if isinstance(chunks, list):
        parts = []
        for item in chunks:
            if isinstance(item, str):
                parts.append(item)
                continue

            page_content = getattr(item, "page_content", None)
            if isinstance(page_content, str):
                parts.append(page_content)
            else:
                parts.append(str(item))
        return "\n\n".join(parts)

    return str(chunks)


def _handle_flight_search(
    query: str,
    llm,
    flight_client: FlightAPIClient,
    retriever,
    chat_history: List[Tuple[str, str]],
    pending_entities: Dict,
    price_compare: bool = False,
) -> Tuple[str, Dict]:
    """Handle flight search and price comparison routing.
    
    Args:
        query: User's input query.
        llm: LLM instance.
        flight_client: FlightAPI flight search client.
        retriever: RAG retriever for policy information.
        chat_history: Conversation history.
        pending_entities: Persisted entities from prior turn.
        price_compare: If True, optimize for price comparison; else general flight search.
    
    Returns:
        (response_text, updated_pending_entities)
    """
    try:
        # Step A: Extract entities from current query
        extracted = extract_entities(query, llm, chat_history)

        # Merge with any pending entities from prior clarification
        if pending_entities:
            merged = merge_entities(pending_entities, extracted)
        else:
            merged = extracted

        # Check if clarification is needed
        needs_clarif, clarif_question = needs_clarification(merged)
        if needs_clarif:
            # Return question and persist entities for next turn
            return (clarif_question, merged)

        # All required fields present — proceed with flight search
        origin = merged.get("origin_iata")
        destination = merged.get("destination_iata")
        departure_date = merged.get("departure_date")
        cabin_class = merged.get("cabin_class", "ECONOMY")
        adults = merged.get("adults", 1)

        # Step B: Call FlightAPI flight search API
        cabin_map = {
            "ECONOMY": "Economy",
            "BUSINESS": "Business",
            "FIRST": "First",
            "PREMIUM_ECONOMY": "Economy",
        }
        cabin = cabin_map.get(cabin_class, "Economy")

        raw = flight_client.search_flights(
            origin=origin,
            destination=destination,
            date=departure_date,
            adults=adults,
            cabin=cabin,
            currency="INR",
        )

        if not raw:
            return (
                "I couldn't fetch live flight data right now. "
                "You can check prices at MakeMyTrip or Goibibo. "
                "I can still answer policy questions.",
                {},
            )

        # Step C: Parse offers from raw FlightAPI data
        offers = parse_flightapi_response(raw, cabin=cabin)

        if not offers:
            return (
                "I couldn't find flights for that route and date. "
                "Please try:\n"
                "- Checking different dates\n"
                "- Verifying the airport codes\n"
                "- Searching directly on MakeMyTrip or Goibibo",
                {},
            )

        # Step D: Normalize currency (pass-through for INR, or convert if needed)
        offers = normalize_currency(offers, target_currency="INR")

        # Step E: Rank and tag offers
        offers = rank_offers(offers)
        offers = tag_offers(offers)

        # Step F: Optionally retrieve policy information
        rag_chunks = None
        policy_keywords = ["baggage", "policy", "refund", "change", "cancellation"]
        if any(keyword in query.lower() for keyword in policy_keywords):
            try:
                rag_chunks = _to_text_chunks(_retrieve_docs(retriever, query))
            except Exception as e:
                logger.warning(f"Failed to retrieve policies: {e}")

        # Step G: Build comprehensive context for LLM
        entities_for_context = {
            "origin": origin,
            "destination": destination,
            "date": departure_date,
            "cabin": cabin_class,
            "adults": adults,
        }

        intent_label = "price_compare" if price_compare else "flight_search"
        context = build_context(
            intent_type=intent_label,
            rag_chunks=rag_chunks,
            flight_offers=offers,
            flight_status=None,
            entities=entities_for_context,
        )

        # Step H: Generate LLM response
        if price_compare:
            system_prompt = (
                "You are an airline pricing specialist. Analyze the flight options provided and focus on price differences. "
                "Explicitly highlight: (1) the cheapest option, (2) the fastest option, (3) the best value option. "
                "Explain what makes each option valuable for different customer needs. Be concise and structured."
            )
        else:
            system_prompt = (
                "You are a helpful airline assistant. Based on the flight search results provided, "
                "help the user find the best flight for their needs. Be friendly and detailed about pricing, "
                "airlines, flight times, and convenience. Recommend options based on their priorities."
            )

        messages = build_messages(system_prompt, context + "\n\nUser query: " + query)
        llm_response = llm.chat(messages, stream=False)

        # Step I: Format offers as markdown table for UI
        table = format_as_markdown_table(offers)

        # Step J: Combine response with table
        final_response = str(llm_response)
        if table:
            final_response += "\n\n" + table

        flight_source = (
            "\n\n---\n**Source:** FlightAPI.io — "
            f"live data fetched at {datetime.now().strftime('%H:%M')} IST"
        )
        final_response += flight_source

        return (final_response, {})

    except Exception as e:
        logger.error(f"Error in flight search: {e}", exc_info=True)
        return (
            "I couldn't search for flights at this moment. Please check your travel details and try again.",
            pending_entities,  # Preserve entities for retry
        )


def _handle_policy(query: str, llm, retriever) -> Tuple[str, Dict]:
    """Handle policy / baggage / cancellation inquiries with RAG.
    
    Args:
        query: User's policy question.
        llm: LLM instance.
        retriever: RAG retriever for policy documents.
    
    Returns:
        (response_text, {})
    """
    try:
        # Retrieve relevant policy documents
        use_hyde = len(query.split()) < 8
        rag_chunks, sources = hyde_retrieve(query, llm, retriever, use_hyde)
        fallback, score = should_fallback(sources)

        if fallback:
            return (build_fallback_response(query, score), {})

        # Build context using context_builder
        context = build_context(
            intent_type="policy",
            rag_chunks=rag_chunks,
            flight_offers=None,
            flight_status=None,
            entities={},
        )

        # Call LLM to generate response
        system_prompt = (
            "You are an airline customer support specialist. Answer the user's question "
            "strictly using the policy information provided. If the answer is not in the provided "
            "information, say 'I do not have that information in the provided documents.' "
            "Always cite your source."
        )

        messages = build_messages(system_prompt, context + "\n\nUser question: " + query)
        response = llm.chat(messages, stream=False)
        response_text = format_inline_citations(str(response), sources)

        return (response_text, {})

    except Exception as e:
        logger.error(f"Error in policy retrieval: {e}", exc_info=True)
        if "Local Ollama model cannot run due to memory limits" in str(e):
            return (str(e), {})
        return (
            "I couldn't retrieve the policy information at this time. Please try again.",
            {},
        )


def _handle_flight_status(
    query: str, llm, aviationstack_client: Optional[AviationstackClient]
) -> Tuple[str, Dict]:
    """Handle flight status inquiries.
    
    Args:
        query: User's flight status query.
        llm: LLM instance.
        aviationstack_client: Aviationstack client for status queries, or None.
    
    Returns:
        (response_text, {})
    """
    try:
        if not aviationstack_client:
            return (
                "Flight status service is not currently available. Please contact support or try again later.",
                {},
            )

        # Step A: Extract flight number from query using regex
        # Pattern: 2 letters (airline code), optional hyphen/space, 2-4 digits
        flight_match = re.search(r"\b([A-Z]{2}[-\s]?\d{2,4})\b", query.upper())

        if not flight_match:
            return (
                "Please provide a flight number to check status (e.g., 6E-204, AI101, SG-156).",
                {},
            )

        flight_number = flight_match.group(1)

        # Step B: Query flight status from Aviationstack
        flight_status = aviationstack_client.get_flight_status(flight_number)

        if flight_status is None:
            return (
                f"I couldn't find live status for flight {flight_number}. "
                f"Please verify the flight number and try again.",
                {},
            )

        # Step C: Build context with flight status
        context = build_context(
            intent_type="flight_status",
            rag_chunks=None,
            flight_offers=None,
            flight_status=flight_status,
            entities={"flight_number": flight_number},
        )

        # Step D: Generate LLM response
        system_prompt = (
            "You are an airline customer support representative providing real-time flight status updates. "
            "Be friendly and clear. Explain the flight status, any delays, and what it means for the passenger."
        )

        messages = build_messages(system_prompt, context + "\n\nUser query: " + query)
        response = llm.chat(messages, stream=False)

        return (str(response), {})

    except Exception as e:
        logger.error(f"Error in flight status lookup: {e}", exc_info=True)
        return (
            "I encountered an error checking flight status. Please try again.",
            {},
        )


def _handle_general(
    query: str, llm, chat_history: List[Tuple[str, str]]
) -> Tuple[str, Dict]:
    """Handle general questions / greetings with no retrieval context."""
    try:
        system_prompt = (
            "You are a friendly airline customer support chatbot. Help the user with general questions, "
            "greetings, and out-of-scope inquiries. Be polite and guide them to relevant services if needed."
        )

        messages = build_messages(system_prompt, query)

        # Add chat history for context
        if chat_history:
            # Insert history before the current user message
            history_messages = [{"role": "system", "content": system_prompt}]
            for user_turn, assistant_turn in chat_history[-3:]:  # Last 3 turns
                history_messages.append({"role": "user", "content": user_turn})
                history_messages.append(
                    {"role": "assistant", "content": assistant_turn}
                )
            history_messages.append({"role": "user", "content": query})
            messages = history_messages

        response = llm.chat(messages, stream=False)
        return (str(response), {})

    except Exception as e:
        logger.error(f"Error in general query handling: {e}", exc_info=True)
        return (
            "I'm not sure how to help with that. Feel free to ask about flights, airline policies, or baggage rules!",
            {},
        )

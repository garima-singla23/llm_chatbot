"""
Flight Offer and Status Formatting.

Formats FlightOffer objects and flight status data into LLM-readable context
and Gradio UI-friendly markdown tables.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from utils.flightapi_parser import FlightOffer


def format_for_llm(offers: List[FlightOffer], query_context: str = "") -> str:
    """
    Format flight offers as a compact text block for LLM context.

    Produces a structured text representation of flight offers that the LLM
    can reason over, including pricing, timing, and tags.

    Args:
        offers: List of FlightOffer objects to format
        query_context: Optional context about the search query (e.g., route, date)

    Returns:
        Formatted string for LLM context, or "No flights found" message
    """
    if not offers:
        return "No flights found for this route and date."

    # Generate timestamp
    current_time = datetime.now().strftime("%H:%M")
    lines = [f"Live flight data fetched at {current_time} IST"]

    if query_context:
        lines.append(f"Search: {query_context}\n")

    # Format up to 5 offers
    for idx, offer in enumerate(offers[:5], 1):
        # Get first and last segment for departure/arrival
        first_segment = offer.segments[0] if offer.segments else None
        last_segment = offer.segments[-1] if offer.segments else None

        if not first_segment or not last_segment:
            continue

        # Extract times (ISO format, get just the time part)
        departure_time = first_segment.departure_time.split("T")[1][:5] if "T" in first_segment.departure_time else first_segment.departure_time
        arrival_time = last_segment.arrival_time.split("T")[1][:5] if "T" in last_segment.arrival_time else last_segment.arrival_time

        # Calculate hours and minutes
        hours = offer.total_duration_minutes // 60
        minutes = offer.total_duration_minutes % 60

        # Format price with commas
        price_str = f"{offer.total_price:,.0f}"

        # Format tags
        tags_str = ", ".join(offer.tags) if offer.tags else "none"

        # Build the option block
        lines.append("---")
        lines.append(
            f"Option {idx}: {first_segment.airline_name} {first_segment.flight_number} | "
            f"{first_segment.departure_iata} → {last_segment.arrival_iata}"
        )
        lines.append(
            f"Price: INR {price_str} | Duration: {hours}h {minutes}m | Stops: {offer.num_stops}"
        )
        lines.append(f"Tags: {tags_str}")
        lines.append(f"Departs: {departure_time} | Arrives: {arrival_time}")

    lines.append("\nSource: FlightAPI.io (live)")

    return "\n".join(lines)


def format_as_markdown_table(offers: List[FlightOffer]) -> str:
    """
    Format flight offers as a markdown table for Gradio UI display.

    Creates a markdown table with flight details. The cheapest option
    is highlighted with bold formatting.

    Args:
        offers: List of FlightOffer objects to format

    Returns:
        Markdown table string, or empty string if no offers
    """
    if not offers:
        return ""

    # Find the cheapest offer for highlighting
    cheapest_idx = min(range(len(offers)), key=lambda i: offers[i].total_price)

    lines = []
    lines.append("| Airline | Flight | Departs | Arrives | Duration | Stops | Price (INR) | Tags |")
    lines.append("|---------|--------|---------|---------|----------|-------|-------------|------|")

    # Format up to 8 offers
    for idx, offer in enumerate(offers[:8]):
        # Get first and last segment
        first_segment = offer.segments[0] if offer.segments else None
        last_segment = offer.segments[-1] if offer.segments else None

        if not first_segment or not last_segment:
            continue

        # Extract times
        departure_time = first_segment.departure_time.split("T")[1][:5] if "T" in first_segment.departure_time else first_segment.departure_time
        arrival_time = last_segment.arrival_time.split("T")[1][:5] if "T" in last_segment.arrival_time else last_segment.arrival_time

        # Calculate duration
        hours = offer.total_duration_minutes // 60
        minutes = offer.total_duration_minutes % 60
        duration = f"{hours}h {minutes}m"

        # Format price
        price = f"{offer.total_price:,.0f}"

        # Format tags
        tags = ", ".join(offer.tags) if offer.tags else "-"

        # Format stops
        stops = str(offer.num_stops)

        # Build row
        row = f"| {first_segment.airline_name} | {first_segment.flight_number} | {departure_time} | {arrival_time} | {duration} | {stops} | {price} | {tags} |"

        # Bold cheapest row
        if idx == cheapest_idx:
            row = f"| **{first_segment.airline_name}** | **{first_segment.flight_number}** | **{departure_time}** | **{arrival_time}** | **{duration}** | **{stops}** | **{price}** | **{tags}** |"

        lines.append(row)

    return "\n".join(lines)


def format_status_for_llm(status: Optional[Dict[str, Any]]) -> str:
    """
    Format flight status as a compact text block for LLM context.

    Takes the parsed flight status dict from AviationstackClient and produces
    a readable summary including current status and delay information.

    Args:
        status: Flight status dictionary from get_flight_status(), or None

    Returns:
        Formatted status string for LLM context
    """
    if status is None:
        return "Flight status unavailable."

    flight_number = status.get("flight_number", "Unknown")
    airline = status.get("airline", "Unknown")
    flight_status = status.get("status", "Unknown")
    scheduled_departure = status.get("scheduled_departure", "Unknown")
    actual_departure = status.get("actual_departure")
    delay_minutes = status.get("delay_minutes")

    lines = []
    lines.append(f"Flight {flight_number} ({airline})")
    lines.append(f"Status: {flight_status}")
    lines.append(f"Scheduled departure: {scheduled_departure}")

    # Handle actual departure and delay
    if actual_departure:
        lines.append(f"Actual departure: {actual_departure}")
    else:
        lines.append("Actual departure: Not yet departed")

    if delay_minutes is not None:
        if delay_minutes == 0:
            lines.append("Delay: No delay reported (on time)")
        elif delay_minutes > 0:
            lines.append(f"Delay: {delay_minutes} minutes late")
        else:
            lines.append(f"Delay: Departed {abs(delay_minutes)} minutes early")
    else:
        lines.append("Delay: Not available")

    return "\n".join(lines)


def format_offer_details(offer: FlightOffer) -> str:
    """
    Format a single flight offer with full itinerary details.

    Useful for displaying detailed information about a selected offer.

    Args:
        offer: FlightOffer object to format

    Returns:
        Formatted string with full offer details
    """
    lines = []

    # Header
    first_segment = offer.segments[0] if offer.segments else None
    last_segment = offer.segments[-1] if offer.segments else None

    if first_segment and last_segment:
        lines.append(f"Flight Offer ID: {offer.offer_id}")
        lines.append(
            f"Route: {first_segment.departure_iata} → {last_segment.arrival_iata}"
        )
    else:
        lines.append(f"Flight Offer ID: {offer.offer_id}")

    # Pricing
    lines.append(f"\nPricing:")
    lines.append(f"  Total: {offer.currency} {offer.total_price:,.2f}")
    lines.append(f"  Base: {offer.currency} {offer.base_price:,.2f}")
    lines.append(f"  Taxes & Fees: {offer.currency} {offer.taxes:,.2f}")

    # Journey summary
    hours = offer.total_duration_minutes // 60
    minutes = offer.total_duration_minutes % 60
    lines.append(f"\nJourney:")
    lines.append(f"  Total Duration: {hours}h {minutes}m")
    lines.append(f"  Stops: {offer.num_stops}")
    lines.append(f"  Segments: {len(offer.segments)}")

    # Segments
    lines.append(f"\nItinerary:")
    for seg_idx, segment in enumerate(offer.segments, 1):
        seg_departure = segment.departure_time.split("T")[1][:5] if "T" in segment.departure_time else segment.departure_time
        seg_arrival = segment.arrival_time.split("T")[1][:5] if "T" in segment.arrival_time else segment.arrival_time
        seg_hours = segment.duration_minutes // 60
        seg_minutes = segment.duration_minutes % 60

        lines.append(f"  Segment {seg_idx}:")
        lines.append(
            f"    {segment.airline_name} {segment.flight_number} | "
            f"{segment.departure_iata} {seg_departure} → {segment.arrival_iata} {seg_arrival}"
        )
        lines.append(f"    Duration: {seg_hours}h {seg_minutes}m")

    # Additional info
    if offer.cabin_class:
        lines.append(f"\nCabin Class: {offer.cabin_class}")

    if offer.seats_remaining is not None:
        lines.append(f"Available Seats: {offer.seats_remaining}")

    if offer.validating_airline:
        lines.append(f"Validating Airline: {offer.validating_airline}")

    # Tags and score
    if offer.tags:
        lines.append(f"Tags: {', '.join(offer.tags)}")

    if offer.value_score > 0:
        lines.append(f"Value Score: {offer.value_score:.2f} / 1.0")

    return "\n".join(lines)

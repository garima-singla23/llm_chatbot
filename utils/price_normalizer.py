"""
Flight Offer Price Normalization and Ranking.

Handles currency normalization, value scoring, and tagging of flight offers.
Produces comparison-ready output sorted by best value.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import requests
from requests.exceptions import RequestException

from utils.flight_parser import FlightOffer

logger = logging.getLogger(__name__)

# Exchange rate cache: {(from_currency, to_currency): (timestamp, rate)}
_exchange_rate_cache: Dict[tuple[str, str], tuple[datetime, float]] = {}
_exchange_rate_ttl = 3600  # 60 minutes


def _get_exchange_rate(from_currency: str, to_currency: str) -> Optional[float]:
    """
    Get exchange rate from cache or fetch from Frankfurter API.

    Args:
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "INR")

    Returns:
        Exchange rate, or None if fetch fails
    """
    # Check cache
    cache_key = (from_currency, to_currency)
    if cache_key in _exchange_rate_cache:
        timestamp, rate = _exchange_rate_cache[cache_key]
        if datetime.now() - timestamp < timedelta(seconds=_exchange_rate_ttl):
            logger.debug(f"Using cached rate {from_currency}/{to_currency}: {rate}")
            return rate
        else:
            del _exchange_rate_cache[cache_key]

    # Same currency, no conversion needed
    if from_currency == to_currency:
        _exchange_rate_cache[cache_key] = (datetime.now(), 1.0)
        return 1.0

    try:
        url = "https://api.frankfurter.app/latest"
        params = {"from": from_currency, "to": to_currency}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        rate = data.get("rates", {}).get(to_currency)

        if rate is not None:
            _exchange_rate_cache[cache_key] = (datetime.now(), rate)
            logger.info(f"Fetched rate {from_currency}/{to_currency}: {rate}")
            return rate
        else:
            logger.warning(f"No rate found for {from_currency}/{to_currency}")
            return None

    except RequestException as e:
        logger.error(f"Failed to fetch exchange rate: {str(e)}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to parse exchange rate response: {str(e)}")
        return None


def normalize_currency(
    offers: List[FlightOffer], target_currency: str = "INR"
) -> List[FlightOffer]:
    """
    Normalize flight offer prices to target currency.

    Amadeus already returns INR when currencyCode=INR is passed in the request,
    so this is mostly a passthrough. However, it fetches live exchange rates
    from Frankfurter API as a fallback for offers in other currencies.

    Args:
        offers: List of flight offers to normalize
        target_currency: Target currency code (default: "INR")

    Returns:
        List of offers with normalized prices and currencies
    """
    normalized = []

    for offer in offers:
        # If already in target currency, keep as is
        if offer.currency == target_currency:
            normalized.append(offer)
            continue

        # Fetch exchange rate
        rate = _get_exchange_rate(offer.currency, target_currency)
        if rate is None:
            logger.warning(
                f"Could not convert {offer.currency} to {target_currency}, "
                f"keeping original currency"
            )
            normalized.append(offer)
            continue

        # Convert prices
        offer.total_price = offer.total_price * rate
        offer.base_price = offer.base_price * rate
        offer.taxes = offer.taxes * rate
        offer.currency = target_currency

        normalized.append(offer)

    return normalized


def rank_offers(offers: List[FlightOffer]) -> List[FlightOffer]:
    """
    Rank flight offers by computed value score.

    Value score combines price, duration, and stops using weighted scoring:
    - price_score: normalized price (0=most expensive, 1=cheapest)
    - time_score: normalized duration (0=longest, 1=shortest)
    - stop_score: 1.0 for nonstop, 0.4 for 1 stop, 0.0 for 2+ stops
    - value_score = 0.5*price + 0.3*time + 0.2*stops

    Args:
        offers: List of flight offers to rank

    Returns:
        List of offers sorted by value_score descending (best value first)
    """
    if not offers:
        return []

    if len(offers) == 1:
        # Single offer: avoid division by zero
        offers[0].value_score = 1.0
        return offers

    # Find min/max for normalization
    prices = [o.total_price for o in offers]
    durations = [o.total_duration_minutes for o in offers]

    min_price = min(prices)
    max_price = max(prices)
    min_duration = min(durations)
    max_duration = max(durations)

    # Calculate value score for each offer
    for offer in offers:
        # Price score: 0 (most expensive) to 1 (cheapest)
        if max_price == min_price:
            price_score = 0.5  # All same price
        else:
            price_score = (max_price - offer.total_price) / (max_price - min_price)

        # Time score: 0 (longest) to 1 (shortest)
        if max_duration == min_duration:
            time_score = 0.5  # All same duration
        else:
            time_score = (max_duration - offer.total_duration_minutes) / (
                max_duration - min_duration
            )

        # Stop score: 1.0 (nonstop) > 0.4 (1 stop) > 0.0 (2+ stops)
        if offer.num_stops == 0:
            stop_score = 1.0
        elif offer.num_stops == 1:
            stop_score = 0.4
        else:
            stop_score = 0.0

        # Weighted value score: price(50%) + time(30%) + stops(20%)
        offer.value_score = (
            0.5 * price_score + 0.3 * time_score + 0.2 * stop_score
        )

    # Sort by value_score descending
    offers.sort(key=lambda x: x.value_score, reverse=True)

    return offers


def tag_offers(ranked: List[FlightOffer]) -> List[FlightOffer]:
    """
    Tag flight offers with relevant metadata.

    Tags applied:
    - "cheapest": offer with minimum total_price
    - "fastest": offer with minimum total_duration_minutes
    - "best value": offer with maximum value_score
    - "direct": offer with no stops (num_stops == 0)

    Args:
        ranked: List of ranked flight offers

    Returns:
        List of tagged offers
    """
    if not ranked:
        return []

    # Find indices for special offers
    cheapest_idx = min(range(len(ranked)), key=lambda i: ranked[i].total_price)
    fastest_idx = min(range(len(ranked)), key=lambda i: ranked[i].total_duration_minutes)
    best_value_idx = max(range(len(ranked)), key=lambda i: ranked[i].value_score)

    # Tag each offer
    for i, offer in enumerate(ranked):
        # Initialize tags list if not present
        if not hasattr(offer, "tags") or offer.tags is None:
            offer.tags = []

        # Add special tags
        if i == cheapest_idx:
            offer.tags.append("cheapest")

        if i == fastest_idx:
            offer.tags.append("fastest")

        if i == best_value_idx:
            offer.tags.append("best value")

        # Add direct flight tag
        if offer.num_stops == 0:
            offer.tags.append("direct")

    return ranked


def get_comparison_summary(
    offers: List[FlightOffer], top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate a summary of top N offers for comparison.

    Args:
        offers: List of ranked and tagged offers
        top_n: Number of top offers to include (default: 3)

    Returns:
        List of simplified offer dictionaries for display
    """
    summary = []

    for offer in offers[:top_n]:
        # Build segments summary
        segments_summary = []
        for seg in offer.segments:
            segments_summary.append(
                {
                    "airline": seg.airline_name,
                    "flight": seg.flight_number,
                    "from": seg.departure_iata,
                    "to": seg.arrival_iata,
                    "departure": seg.departure_time,
                    "arrival": seg.arrival_time,
                }
            )

        # Format duration
        hours = offer.total_duration_minutes // 60
        minutes = offer.total_duration_minutes % 60
        duration_display = f"{hours}h {minutes}m"

        summary.append(
            {
                "offer_id": offer.offer_id,
                "price": f"{offer.currency} {offer.total_price:.2f}",
                "base_price": f"{offer.currency} {offer.base_price:.2f}",
                "taxes": f"{offer.currency} {offer.taxes:.2f}",
                "cabin": offer.cabin_class,
                "duration": duration_display,
                "duration_minutes": offer.total_duration_minutes,
                "stops": offer.num_stops,
                "value_score": f"{offer.value_score:.2f}",
                "tags": offer.tags,
                "segments": segments_summary,
                "validating_airline": offer.validating_airline,
                "seats_remaining": offer.seats_remaining,
            }
        )

    return summary

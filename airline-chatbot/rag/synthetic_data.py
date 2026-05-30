import datetime
import json
from pathlib import Path


POLICIES = {
    ("indigo", "baggage"): """
IndiGo baggage policy guide for domestic and international travelers.

Cabin baggage allowance:
Passengers traveling on standard IndiGo fares may carry one cabin bag up to 7 kg. The bag must fit within 55 x 35 x 25 cm, including wheels and handles. In addition to this, one small personal item such as a laptop bag or ladies purse is usually permitted when it can be placed under the seat and does not obstruct safety access.

Checked baggage allowance:
For domestic economy tickets, the standard free checked baggage allowance is 15 kg. For business-style bundled fares where available, the allowance is typically 25 kg. On international routes, allowance may vary by fare family and bilateral route rules, so travelers should verify route-specific conditions during booking.

Excess baggage charges:
If a passenger exceeds the free checked baggage allowance, excess fees are charged by slab. For the first 3 kg over allowance, a common reference rate is INR 350 per kg. Beyond 3 kg, a higher rate around INR 500 per kg may apply. Charges at the airport are generally higher than pre-purchased add-on baggage online. Travelers are advised to pre-book baggage to reduce cost and avoid queue delays.

Sports equipment and special baggage:
Sports equipment such as golf sets, skis, cricket kits, diving kits, and bicycles may be accepted as checked baggage if packed properly. Oversized or fragile sports items can attract additional handling fees. Batteries in e-bikes, hoverboards, and similar devices are typically restricted due to lithium battery safety regulations.

Oversized baggage:
Items that exceed airline dimension limits for standard checked luggage may be classified as oversized baggage. Such items are accepted only if operationally feasible and may require an oversize fee in addition to excess weight charges.

Prohibited items:
Dangerous goods are not allowed in checked or cabin baggage where prohibited by aviation security regulations. This generally includes explosives, flammables, corrosives, compressed gases, and toxic substances. Power banks and spare lithium batteries are usually allowed only in cabin baggage and not in checked baggage. Sharp objects, tools, and sporting blades are typically not allowed in cabin baggage.

Packing and compliance:
Each bag should be properly tagged with name and contact details. Locks should comply with airport screening requirements. If baggage appears damaged, inadequately packed, or unsafe for loading, check-in staff may refuse acceptance until repacking is completed.

Passenger advisory:
For faster boarding, passengers should keep medicines, valuables, travel documents, chargers, and essentials in cabin baggage. Important items should never be packed in checked baggage. Policy conditions may change based on route, season, regulator advisories, and fare type. Passengers should cross-check with official airline communication before travel.

Synthetic explanatory appendix:
This synthetic policy is intended for fallback RAG behavior and training support. It models realistic airline baggage communication style by combining allowance rules, charge slabs, safety restrictions, and practical check-in guidance. The text should be treated as informational content only and not a legal fare contract. Any baggage dispute, waiver request, or compensation claim must be handled according to the latest published airline tariff, civil aviation rules, and the passenger itinerary conditions.
""",
    ("indigo", "refund"): """
IndiGo cancellation and refund policy reference text.

24-hour cancellation window:
If a booking is canceled within 24 hours of purchase and the scheduled departure is sufficiently far away, passengers may be eligible for reduced cancellation charges or partial refund under airline and regulatory standards. Eligibility depends on route, fare category, and booking channel.

7-day tier rule:
For cancellations made more than 7 days before departure, a representative cancellation fee of INR 3000 per passenger may apply for certain fare types. Promo fares can carry stricter conditions.

3-day tier rule:
For cancellations made between 3 and 7 days before departure, a representative cancellation fee of INR 3500 may apply. The final refunded amount depends on base fare, taxes, and applicable surcharges.

Less than 3 days before departure:
For cancellations made less than 3 days before departure, cancellation penalties are generally higher. A representative structure is INR 4000 plus applicable fuel surcharge adjustments. In many low-fare tickets, the residual refundable amount may be minimal.

No-show policy:
If the passenger does not report for check-in before cutoff and does not cancel in advance, the booking may be treated as no-show and may be non-refundable except for eligible taxes where required by law.

Refund timeline:
Approved refunds are commonly processed within 7 to 10 business days for card payments. Wallet and UPI timelines may vary by issuer and settlement gateway. Agency bookings can take longer depending on intermediary reconciliation.

Channel-specific handling:
Bookings made through travel agents or online aggregators are often required to be canceled through the same channel. Airline support teams may have limited ability to override agency workflows unless there is an operational disruption.

Disruption-related waivers:
In case of schedule changes, cancellation by airline, severe weather, government restrictions, or extraordinary operational disruptions, waiver options such as free rebooking, travel credits, or full refund may be offered according to the disruption policy in force.

Refund components:
Refund computations usually consider fare class rules, discount conditions, convenience fees, ancillaries, seat selection, meals, and baggage add-ons. Some ancillaries may be non-refundable once consumed or marked as fulfilled.

Claim documentation:
Passengers should keep PNR, booking invoice, payment proof, and communication logs. For unresolved claims, escalation through airline grievance channels and regulator complaint portals may be necessary.

Synthetic explanatory appendix:
This synthetic text is designed to emulate realistic refund rule language for fallback knowledge generation in RAG workflows. It intentionally includes tiered penalties, no-show treatment, and processing timelines to support user intent handling for cancellation questions. Always verify current policy directly from official carrier notices before presenting final customer guidance.
""",
    ("indigo", "checkin"): """
IndiGo check-in and boarding policy guide.

Web check-in window:
Online check-in typically opens 48 hours before scheduled departure and closes 60 minutes before departure for most domestic routes. For selected international or high-regulation routes, check-in timing may differ and passengers should verify trip-specific instructions.

Airport check-in window:
Passengers are generally advised to reach the airport at least 3 hours before departure for domestic and even earlier for international journeys. Airport check-in counters commonly close 45 minutes before domestic departure, subject to airport operations.

Self-service kiosks:
At larger airports, self-service kiosks are available for eligible itineraries. Kiosks support seat assignment, boarding pass printing, and sometimes baggage tag printing. Kiosk services may be unavailable for infants, special assistance cases, and some security-sensitive routes.

Accepted travel documents:
Commonly accepted ID proofs include Aadhaar, Passport, PAN card, Voter ID, and Driving License, subject to current security regulations. For international travel, valid passport and visa documents are mandatory. Name on ticket must match identity document.

Baggage drop and security sequence:
Passengers who complete web check-in with checked baggage still need to report at baggage drop before closure. After baggage drop, passengers must clear security and reach boarding gate before final call.

Gate closure policy:
Boarding gates generally close before departure as per airport and airline operations. A representative closure time is around 25 minutes before departure, but operational announcements can vary. Late arrival at the gate can lead to denied boarding without refund entitlement under strict fare rules.

Special categories:
Passengers requiring wheelchair support, unaccompanied minor handling, medical clearance, or oversized baggage screening should report early to allow additional processing. Some categories are not eligible for purely digital check-in completion.

Re-issuance and missed flights:
If a passenger misses check-in cutoff, rebooking options are usually governed by fare conditions and applicable change penalties. Flex fares may allow partial waiver while promotional fares may require fresh ticket purchase.

Operational advisories:
Airport congestion, weather events, and security queues can significantly increase transit times. Passengers should monitor SMS and email updates, ensure mobile numbers are reachable, and keep the boarding pass readily available for repeated checkpoints.

Synthetic explanatory appendix:
This fallback policy text is created for synthetic knowledge coverage in RAG systems, especially for check-in and gate closure questions. It is written in practical passenger language to reflect common airline guidance and airport procedures while remaining non-contractual. Users should verify exact timings and document rules from official airline and airport notices before travel.
""",
    ("air_india", "baggage"): """
Air India baggage policy reference.

Domestic baggage allowance:
For many domestic economy fares, a standard checked baggage allowance of around 15 kg is common. Premium cabins may receive higher allowances, and certain bundled products can allow up to 25 kg depending on fare family.

International baggage allowance:
For many international economy routes, one checked piece up to 23 kg is a common benchmark under piece concept. Business class frequently allows one or more pieces with higher per-piece limits, often up to 32 kg each based on route and fare rules.

Hand baggage:
A typical hand baggage allowance is around 8 kg for eligible cabins, with dimension and safety restrictions. Small personal items may be permitted in addition when compliant with security rules.

Infant allowance:
Infants traveling without a separate seat may receive limited checked baggage allowance and one collapsible stroller or baby carry item subject to handling feasibility and route policy.

Excess baggage charges:
Excess fees are charged when passengers exceed free allowance by weight or piece count. Charges differ by sector and may be substantially higher at the airport than pre-purchased add-on rates.

Sports equipment and special baggage:
Sports equipment may be accepted subject to declaration, packaging, and dimension limits. Fragile articles and musical instruments can require additional protective packing and may be accepted on limited liability terms.

Restricted and prohibited items:
Dangerous goods restrictions apply to all baggage. Lithium battery devices, aerosols, and sharp tools are regulated by cabin-vs-check rules under aviation safety norms.

Through-check and interline:
For itineraries involving partner carriers, the most significant carrier rule or interline agreement can affect allowance. Passengers should verify baggage policy for all sectors, not just the first flight.

Claims and liability:
Delayed, lost, or damaged baggage claims should be filed at arrival with a Property Irregularity Report where applicable. Compensation follows applicable conventions, domestic law, and declared value terms.

Synthetic explanatory appendix:
This synthetic Air India baggage narrative is intended for fallback chatbot responses when source scraping is unavailable. It emphasizes realistic passenger concerns: domestic/international allowances, hand baggage rules, infant coverage, excess costs, sports equipment treatment, and claim pathways. Final advice should always be validated against current official airline tariff publications.
""",
    ("spicejet", "baggage"): """
SpiceJet baggage and check-in policy reference.

Domestic cabin baggage:
Passengers are generally allowed one cabin bag up to 7 kg, subject to size limits and security checks. Personal item allowances may depend on airport enforcement conditions.

Checked baggage allowance:
A common domestic free checked baggage allowance is around 15 kg for standard economy fares. Add-on baggage options can be purchased during booking, manage-booking flows, or at the airport.

SpiceMax and bundled benefits:
SpiceMax or equivalent premium products may include additional comfort and selected ancillary benefits, which can include expanded baggage allowance depending on campaign and fare terms.

Excess baggage charges:
When checked baggage exceeds allowance, excess fees apply per kilogram. Airport pricing can be significantly higher than pre-booked rates. Overweight and oversize baggage may both trigger additional fees.

Check-in timings:
Web check-in and airport check-in windows are governed by route and airport operations. Passengers should arrive early to complete baggage drop and security formalities, especially during peak travel hours.

Prohibited and restricted articles:
The airline enforces aviation security norms for hazardous and prohibited goods. Power banks are usually carried in cabin baggage only, and sharp or restricted objects are disallowed in hand baggage.

Special handling items:
Sports gear, musical instruments, and fragile items may need declaration, protective packaging, and handling fees. Some items are accepted subject to operational space and loading limits.

No-show and baggage impact:
If a passenger fails to report in time and is marked no-show, baggage allowances and ancillary benefits are not carried forward automatically and may require reissue under fresh fare conditions.

Passenger advisory:
Travelers should keep essential medicines, IDs, and valuables in cabin baggage. Baggage tags should include phone and email details to reduce recovery delays in case of mishandling.

Synthetic explanatory appendix:
This fallback policy text is written to support RAG coverage for SpiceJet baggage queries, including baseline 7 kg cabin and 15 kg checked references, SpiceMax mention, excess pricing behavior, and check-in timing emphasis. Since rules can change by fare and season, official airline pages should be used as the final source of truth.
""",
    ("vistara", "baggage"): """
Vistara baggage and travel benefits policy guide.

Economy allowance:
For domestic economy fares, a representative checked baggage allowance is around 15 kg. On international economy routes, allowance often follows piece concept with approximately 23 kg per checked bag for eligible fares.

Premium Economy allowance:
Premium Economy fares commonly provide higher checked baggage entitlement, often around 25 kg for domestic travel where weight concept applies.

Business Class allowance:
Business class passengers generally receive substantially higher allowance, often around 32 kg per checked piece on eligible international sectors under piece concept rules.

Cabin baggage:
Cabin baggage limits apply by cabin type and route, with strict dimension and safety screening compliance. Personal items may be allowed if compact and security-compliant.

Club Vistara and upgrades:
Frequent flyers may use Club Vistara points or milestone benefits for upgrades, and in certain circumstances this can improve baggage entitlement if the upgraded cabin benefits are applied as per program terms.

Excess and prepaid baggage:
Passengers exceeding allowance must pay excess charges. Prepaid baggage options are generally more economical than airport charges. Charges can vary by route, connection pattern, and fare class.

Special and sports baggage:
Sports and special equipment are accepted subject to packaging, dimensions, and aircraft hold limitations. Fragile or high-value items should be appropriately packed and declared where needed.

Infant and family travel:
Infant and family allowances may include specific provisions for stroller acceptance and baby essentials, depending on route policy and aircraft handling capability.

Irregular operations and baggage support:
In case of delayed or mishandled baggage, passengers should report at arrival help desks and retain all journey documents. Resolution and compensation depend on prevailing aviation regulations and carrier policies.

Synthetic explanatory appendix:
This synthetic Vistara baggage policy text is intended as a realistic fallback for chatbot response generation, especially for users asking domestic vs international allowance differences and cabin upgrade effects. It includes Club Vistara upgrade context and class-based allowance structures. Before customer communication, confirm against official current policy notices.
""",
}


def _ensure_min_words(text: str, min_words: int = 500) -> str:
    words = text.split()
    if len(words) >= min_words:
        return text.strip()

    filler = (
        "This synthetic section provides additional explanatory guidance on policy usage, "
        "fare conditions, airport processing, passenger obligations, safety restrictions, "
        "and dispute handling workflows. Travelers should review ticket terms, airline notices, "
        "and regulator advisories before relying on any single summary. "
    )

    pieces = [text.strip()]
    while len(" ".join(pieces).split()) < min_words:
        pieces.append(filler)

    return "\n\n".join(pieces).strip()


def generate_synthetic(output_dir: str = "data/raw") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_path.resolve()}")

    for (airline, topic), policy_text in POLICIES.items():
        file_name = f"{airline}_{topic}_synthetic.txt"
        txt_path = output_path / file_name
        meta_path = txt_path.with_suffix(".meta.json")

        if txt_path.exists():
            print(f"[SKIP] {txt_path.name} already exists")
            continue

        body = _ensure_min_words(policy_text, min_words=500)
        full_text = (
            "# SYNTHETIC DATA - verify against official source\n\n"
            f"{body}\n"
        )

        txt_path.write_text(full_text, encoding="utf-8")

        meta = {
            "source_type": "synthetic",
            "airline": airline,
            "topic": topic,
            "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[OK] Saved {txt_path.name} and {meta_path.name}")


if __name__ == "__main__":
    generate_synthetic()

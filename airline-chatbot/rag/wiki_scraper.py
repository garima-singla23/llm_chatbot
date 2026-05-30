import json
import time
from pathlib import Path
import datetime
import requests


WIKI_PAGES = {
    "indigo": [
        "IndiGo",
        "IndiGo_6E_frequent_flyer",
    ],
    "air_india": [
        "Air_India",
        "Air_India_Express",
    ],
    "spicejet": [
        "SpiceJet",
    ],
    "vistara": [
        "Vistara",
    ],
    "general": [
        "Baggage_allowance",
        "Airline_ticket",
        "Air_travel_disruption",
        "DGCA",
        "Montreal_Convention",
        "EU_261/2004",
    ],
}


# Regulations and international rules useful for refunds/compensation/liability
WIKI_PAGES.setdefault("regulations", [
    "EC_261/2004",
    "Warsaw_Convention",
    "Montreal_Convention",
    "Directorate_General_of_Civil_Aviation_(India)",
    "Civil_Aviation_Requirements",
    "Passenger_rights",
])


WIKI_FULLTEXT_URL = "https://en.wikipedia.org/w/api.php"


def fetch_wiki_page(title: str) -> str:
    """Fetch the full plaintext extract for a Wikipedia page title.

    Returns the extract string, or empty string on failure.
    Adds a short delay to be polite.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
    }

    headers = {"User-Agent": "AirAssistBot/1.0 (https://example.com)"}

    try:
        resp = requests.get(WIKI_FULLTEXT_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return ""

        # pages is a dict keyed by pageid; get first page
        for page in pages.values():
            extract = page.get("extract", "")
            # polite delay
            time.sleep(0.5)
            return extract or ""

        time.sleep(0.5)
        return ""

    except Exception:
        try:
            time.sleep(0.5)
        except Exception:
            pass
        return ""


def scrape_wiki(output_dir: str = "data/raw") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"[INFO] Output directory: {out_path.resolve()}")

    for airline, titles in WIKI_PAGES.items():
        print(f"[INFO] Scraping wiki pages for {airline} ({len(titles)} pages)")
        for idx, title in enumerate(titles, start=1):
            print(f"  [INFO] ({airline} {idx}/{len(titles)}) Fetching: {title}")
            text = fetch_wiki_page(title)

            file_name = f"{airline}_{idx}_wiki.txt"
            txt_path = out_path / file_name
            txt_path.write_text(text, encoding="utf-8")

            source_url = f"https://en.wikipedia.org/wiki/{title}"
            # For regulation pages store them as general/regulation in metadata
            if airline == "regulations":
                meta_airline = "general"
                meta_source_type = "regulation"
            else:
                meta_airline = airline
                meta_source_type = "wikipedia"

            meta = {
                "airline": meta_airline,
                "source_url": source_url,
                "source_type": meta_source_type,
                "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "title": title,
            }
            meta_path = txt_path.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            total += 1
            print(f"  [OK] Saved {txt_path.name} ({len(text):,} chars)")

    print(f"[INFO] Completed. Total pages processed: {total}")


if __name__ == "__main__":
    scrape_wiki()

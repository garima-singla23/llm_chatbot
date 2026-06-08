import datetime
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


PLAYWRIGHT_SOURCES = [
    {
        "url": "https://www.iata.org/en/programs/passenger/passenger-rights/",
        "airline": "general",
        "topic": "passenger_rights",
        "filename": "iata_passenger_rights",
    },
    {
        "url": "https://www.iata.org/en/programs/passenger/baggage/",
        "airline": "general",
        "topic": "baggage",
        "filename": "iata_baggage",
    },
    {
        "url": "https://dgca.gov.in/digigov-portal/?page=jsp/dgca/InventoryList/passengercharter.jsp",
        "airline": "general",
        "topic": "passenger_rights",
        "filename": "dgca_charter",
    },
    {
        "url": "https://www.airportsindia.com/passenger-information",
        "airline": "general",
        "topic": "airport",
        "filename": "aai_passenger_info",
    },
]


def scrape_page_playwright(url: str) -> str:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")

            for selector in ["nav", "footer", "header", "script", "style"]:
                page.eval_on_selector_all(selector, "elements => elements.forEach(el => el.remove())")

            text = page.inner_text("body")
            browser.close()
            return text
    except Exception as exc:
        print(f"[ERROR] Failed to scrape {url}: {exc}")
        return ""


def scrape_playwright(output_dir: str = "data/raw") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output directory: {output_path.resolve()}")

    for source in PLAYWRIGHT_SOURCES:
        url = source["url"]
        airline = source["airline"]
        topic = source["topic"]
        filename = source["filename"]

        txt_path = output_path / f"{filename}_playwright.txt"
        meta_path = txt_path.with_suffix(".meta.json")

        if txt_path.exists():
            print(f"[SKIP] {txt_path.name} already exists")
            continue

        print(f"[INFO] Scraping: {url}")
        text = scrape_page_playwright(url)

        txt_path.write_text(text, encoding="utf-8")

        meta = {
            "airline": airline,
            "source_url": url,
            "source_type": "playwright",
            "topic": topic,
            "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "filename": filename,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"[OK] Saved {txt_path.name} and {meta_path.name}")


if __name__ == "__main__":
    scrape_playwright()

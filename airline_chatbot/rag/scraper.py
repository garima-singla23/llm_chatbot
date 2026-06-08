from pathlib import Path
import json
import datetime
import requests
from playwright.sync_api import sync_playwright


AIRLINE_SOURCES = {
    "indigo": [
        "https://www.goindigo.in/baggage.html",
        "https://www.goindigo.in/faq.html",
        "https://www.goindigo.in/refund-process.html",
    ],
    "air_india": [
        "https://www.airindia.com/in/en/travel-information/baggage-guidelines.html",
        "https://www.airindia.com/in/en/help/faq.html",
        "https://www.airindia.com/in/en/manage-booking/cancellation-and-refund.html",
    ],
    "spicejet": [
        "https://www.spicejet.com/BaggagePolicy.aspx",
        "https://www.spicejet.com/FAQ.aspx",
        "https://www.spicejet.com/tnc.aspx?cat=13",
    ],
    "vistara": [
        "https://www.airvistara.com/in/en/travel-information/baggage",
        "https://www.airvistara.com/in/en/faq",
        "https://www.airvistara.com/in/en/travel-information/cancellations-and-refunds",
    ],
}


def scrape_page_playwright(url: str, timeout: int = 60000) -> str:
    """
    Fetches a URL using a real Chromium browser,
    waits for the network to settle, then extracts visible text.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-IN",
        )

        page = context.new_page()

        try:
            # networkidle waits until no network requests for 500ms
            page.goto(url, wait_until="networkidle", timeout=timeout)

            # Extra wait for lazy-loaded content
            page.wait_for_timeout(3000)

            # Remove noisy elements before extracting text
            page.evaluate("""
                () => {
                    const selectors = [
                        'nav', 'footer', 'script', 'style',
                        'header', '.cookie-banner', '#cookie-notice',
                        '.popup', '.modal', '.overlay'
                    ];
                    selectors.forEach(sel => {
                        document.querySelectorAll(sel)
                            .forEach(el => el.remove());
                    });
                }
            """)

            # Extract clean text
            text = page.inner_text("body")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)

        except Exception as exc:
            print(f"[ERROR] Playwright failed on {url}: {exc}")
            return ""

        finally:
            browser.close()


def debug_response(url):
    """Debug helper to inspect raw HTTP response from a URL using requests."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status Code : {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Body length : {len(response.text)} chars")
        print(f"First 500 chars:\n{response.text[:500]}\n")
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}\n")


def scrape_all(output_dir: str = "data/raw") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    print(f"[INFO] Output directory: {output_path.resolve()}")

    for airline, urls in AIRLINE_SOURCES.items():
        print(f"\n[INFO] Scraping '{airline}' ({len(urls)} pages) ...")

        for index, url in enumerate(urls, start=1):
            print(f"  [{airline}  {index}/{len(urls)}] {url}")

            text = scrape_page_playwright(url)

            if not text:
                print(f"  [WARN] Empty content returned for {url}")

            # ── Save text ──────────────────────────────────────────
            file_name = f"{airline}_{index}.txt"
            txt_path = output_path / file_name
            txt_path.write_text(text, encoding="utf-8")

            # ── Save metadata ──────────────────────────────────────
            meta = {
                "airline": airline,
                "source_url": url,
                "file": file_name,
                "char_count": len(text),
                "scraped_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            meta_path = txt_path.with_suffix(".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            total_saved += 1
            print(f"  [OK]  Saved {txt_path.name}  ({len(text):,} chars)")

    print(f"\n[INFO] Done. Total pages saved: {total_saved}")


if __name__ == "__main__":
    # Uncomment below to debug a single URL before running full scrape
    # debug_response("https://www.goindigo.in/baggage.html")
    
    scrape_all()
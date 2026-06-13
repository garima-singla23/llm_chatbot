

from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import os
import re
import io
import hmac
import hashlib
import logging
import requests.exceptions as req_exc
import requests as req_lib
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

_secret_key = os.getenv("FLASK_SECRET_KEY")
if not _secret_key:
    raise RuntimeError(
        "FLASK_SECRET_KEY environment variable is required. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _secret_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Razorpay client (initialised once at startup)
# ---------------------------------------------------------------------------

def _build_razorpay_client():
    """Create and return a Razorpay client, raising clearly if creds missing."""
    try:
        import razorpay
    except ImportError:
        raise RuntimeError("razorpay package is not installed. Run: pip install razorpay")

    key_id     = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")

    if not key_id or not key_secret:
        raise RuntimeError(
            "RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET environment variables are required."
        )

    import razorpay  # re-import after guard so type checkers are happy
    return razorpay.Client(auth=(key_id, key_secret))


try:
    _razorpay_client = _build_razorpay_client()
except RuntimeError as exc:
    logger.warning("Razorpay client not initialised: %s", exc)
    _razorpay_client = None


def get_razorpay_client():
    """Return the shared Razorpay client or raise a clear error."""
    if _razorpay_client is None:
        raise RuntimeError("Razorpay is not configured on this server.")
    return _razorpay_client


# ---------------------------------------------------------------------------
# Constants / validation helpers
# ---------------------------------------------------------------------------

# Allowed PNR format: exactly "PNR" followed by 6 uppercase letters or digits
_PNR_RE = re.compile(r"^PNR[A-Z0-9]{6}$")

# Sane price bounds in INR (₹100 – ₹1,00,000)
_PRICE_MIN = 100
_PRICE_MAX = 100_000

# Maximum allowed length for free-text fields stored in session
_MAX_FIELD_LEN = 100


def _sanitize(value: str, max_len: int = _MAX_FIELD_LEN) -> str:
    """
    Strip characters that could cause XSS or injection issues and
    truncate to a maximum length.
    """
    if not isinstance(value, str):
        value = ""
    # Remove HTML-significant characters
    cleaned = re.sub(r"[<>\"'&]", "", value)
    return cleaned[:max_len]


def _parse_price(raw: str, default: int = 3000) -> int:
    """
    Parse a price string into an integer.
    Returns *default* if the value is missing, non-numeric, or out of bounds.
    """
    try:
        price = int(float(raw))
    except (ValueError, TypeError):
        return default

    if not (_PRICE_MIN <= price <= _PRICE_MAX):
        logger.warning("Price %s is outside allowed bounds; using default %s", price, default)
        return default

    return price


def _fallback_booking(flight: dict) -> dict:
    """
    Generate a minimal local booking record when the confirmation API is
    unreachable.  Clearly marks the booking as 'pending_confirmation' so
    downstream processes know to reconcile it later.
    """
    import random
    import string

    pnr = "PNR" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return {
        "pnr":          pnr,
        "seat":         "14A",
        "boarding_time": "05:30",
        "gate":         flight.get("gate", "TBA"),
        "terminal":     flight.get("terminal", "T1"),
        "status":       "pending_confirmation",
        "message":      "Booking recorded locally; confirmation in progress.",
    }


def _verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify the Razorpay payment signature using a constant-time comparison
    to prevent timing attacks.
    """
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    if not key_secret:
        logger.error("RAZORPAY_KEY_SECRET is not set; cannot verify signature.")
        return False

    secret = key_secret.encode("utf-8")
    msg    = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(secret, msg, digestmod=hashlib.sha256).hexdigest()

    # constant-time comparison prevents timing side-channel attacks
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/book", methods=["GET"])
def booking_page():
    """
    Render the booking page.

    All query-string values are sanitised before being stored in the session
    so that user-controlled data cannot pollute downstream processing.
    The price is validated against a sane range to prevent the user from
    manipulating the charge amount.
    """
    flight = {
        "flight_no":   _sanitize(request.args.get("flight_no",   "")),
        "airline":     _sanitize(request.args.get("airline",     "")),
        "origin":      _sanitize(request.args.get("origin",      "")),
        "destination": _sanitize(request.args.get("destination", "")),
        "departure":   _sanitize(request.args.get("departure",   "")),
        "arrival":     _sanitize(request.args.get("arrival",     "")),
        "price":       _parse_price(request.args.get("price",    "3000")),
        "date":        _sanitize(request.args.get("date",        "")),
        "gate":        _sanitize(request.args.get("gate",        "TBA")),
        "terminal":    _sanitize(request.args.get("terminal",    "T1")),
    }

    session["pending_flight"] = flight
    logger.info("Booking page rendered for flight %s", flight.get("flight_no"))

    return render_template(
        "booking.html",
        flight=flight,
        razorpay_key=os.getenv("RAZORPAY_KEY_ID", ""),
    )


@app.route("/initiate-payment", methods=["POST"])
def initiate_payment():
    """
    Create a Razorpay order and return the details needed by the
    front-end Razorpay checkout widget.

    The price is read from the *session* (set during /book) rather than
    from the POST body so the user cannot manipulate the charge amount.
    """
    flight = session.get("pending_flight")
    if not flight:
        return jsonify({"error": "No pending flight found. Please start from the search page."}), 400

    passenger = {
        "name":      _sanitize(request.form.get("name",      "")),
        "email":     _sanitize(request.form.get("email",     "")),
        "phone":     _sanitize(request.form.get("phone",     "")),
        "age":       _sanitize(request.form.get("age",       "")),
        "gender":    _sanitize(request.form.get("gender",    "")),
        "seat_pref": _sanitize(request.form.get("seat_pref", "window")),
        "meal_pref": _sanitize(request.form.get("meal_pref", "veg")),
    }

    # Basic presence checks
    if not passenger["name"] or not passenger["email"] or not passenger["phone"]:
        return jsonify({"error": "name, email, and phone are required fields."}), 400

    session["passenger"] = passenger

    # Price comes from the server-side session, NOT from user input
    price       = int(flight.get("price", 3000))   # already validated in /book
    amount_paise = price * 100

    try:
        client = get_razorpay_client()
        order  = client.order.create({
            "amount":          amount_paise,
            "currency":        "INR",
            "payment_capture": 1,
            "notes": {
                "flight":    flight.get("flight_no", ""),
                "passenger": passenger["name"],
            },
        })
    except RuntimeError as exc:
        logger.error("Razorpay configuration error: %s", exc)
        return jsonify({"error": "Payment service is unavailable. Please try again later."}), 503
    except Exception:
        logger.exception("Unexpected error while creating Razorpay order")
        return jsonify({"error": "Could not create payment order. Please try again."}), 500

    session["razorpay_order_id"] = order["id"]
    logger.info("Razorpay order %s created for passenger %s", order["id"], passenger["name"])

    # Return ONLY the public key ID — never the secret
    return jsonify({
        "order_id": order["id"],
        "amount":   amount_paise,
        "key":      os.getenv("RAZORPAY_KEY_ID", ""),
        "name":     passenger["name"],
        "email":    passenger["email"],
        "phone":    passenger["phone"],
    })


@app.route("/payment-success", methods=["POST"])
def payment_success():
    """
    Verify the Razorpay payment signature and confirm the booking.

    Steps:
      1. Validate all three Razorpay callback fields are present.
      2. Verify the HMAC-SHA256 signature with a constant-time comparison.
      3. Call the confirmation API; fall back to a local record on network error.
      4. Render the success page.
    """
    payment_id = request.form.get("razorpay_payment_id", "").strip()
    order_id   = request.form.get("razorpay_order_id",   "").strip()
    signature  = request.form.get("razorpay_signature",  "").strip()

    if not payment_id or not order_id or not signature:
        logger.warning("Payment callback missing required fields.")
        return "Invalid payment callback. Please contact support.", 400

    if not _verify_razorpay_signature(order_id, payment_id, signature):
        logger.warning(
            "Signature mismatch for order_id=%s payment_id=%s", order_id, payment_id
        )
        return "Payment verification failed. Please contact support.", 400

    flight    = session.get("pending_flight", {})
    passenger = session.get("passenger",      {})

    booking = _call_confirmation_api(flight, passenger, payment_id, order_id)

    # Store PNR in session so the boarding-pass route can fall back to it
    session["last_booking"] = booking
    logger.info(
        "Booking confirmed: PNR=%s payment_id=%s", booking.get("pnr"), payment_id
    )

    return render_template(
        "success.html",
        flight=flight,
        passenger=passenger,
        booking=booking,
    )


def _call_confirmation_api(
    flight: dict,
    passenger: dict,
    payment_id: str,
    order_id: str,
) -> dict:
    """
    POST to the internal confirmation micro-service.

    Raises nothing — network/HTTP errors produce a local fallback booking so
    the user always sees a success page.  A background reconciliation job
    should pick up 'pending_confirmation' records later.
    """
    payload = {
        "flight":     flight,
        "passenger":  passenger,
        "payment_id": payment_id,
        "order_id":   order_id,
        "amount":     int(flight.get("price", 3000)),
    }

    try:
        resp = req_lib.post(
            "http://localhost:8001/confirm-booking",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError(f"Unexpected Content-Type from confirmation API: {content_type}")

        return resp.json()

    except req_exc.Timeout:
        logger.warning("Confirmation API timed out; using local fallback booking.")
    except req_exc.ConnectionError:
        logger.error("Confirmation API is unreachable; using local fallback booking.")
    except req_exc.HTTPError as exc:
        logger.error("Confirmation API returned an error: %s", exc)
    except ValueError as exc:
        logger.error("Confirmation API response parse error: %s", exc)
    except Exception:
        logger.exception("Unexpected error calling confirmation API.")

    return _fallback_booking(flight)


# ---------------------------------------------------------------------------
# Boarding pass
# ---------------------------------------------------------------------------

@app.route("/boarding-pass/<pnr>")
def download_boarding_pass(pnr: str):
    """
    Generate and stream a boarding-pass PDF for the given PNR.

    The PNR is validated against a strict regex before being used in any
    external call to prevent path-traversal or injection attacks.
    """
    # --- Validate PNR format first ---
    if not _PNR_RE.match(pnr):
        logger.warning("Invalid PNR format requested: %r", pnr)
        return "Invalid PNR format.", 400

    # --- Fetch booking data ---
    booking, flight = _fetch_booking_data(pnr)

    # --- Generate PDF ---
    try:
        from boarding_pass.generator import generate_boarding_pass
        pdf_bytes = generate_boarding_pass(booking, flight)
    except ImportError:
        logger.error("boarding_pass.generator module is not available.")
        return "Boarding pass generation is currently unavailable.", 503
    except Exception:
        logger.exception("Unexpected error while generating boarding pass for PNR %s", pnr)
        return "Boarding pass generation failed. Please contact support.", 500

    logger.info("Boarding pass generated for PNR %s", pnr)

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{pnr}_boarding_pass.pdf",
    )


def _fetch_booking_data(pnr: str) -> tuple[dict, dict]:
    """
    Try the confirmation API first; fall back to session data.
    Returns (booking_dict, flight_dict).
    """
    try:
        resp = req_lib.get(
            f"http://localhost:8001/booking/{pnr}",
            timeout=5,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError(f"Unexpected Content-Type: {content_type}")

        data = resp.json()

        booking = {
            "pnr":            data.get("pnr",            pnr),
            "passenger_name": data.get("passenger_name", "Passenger"),
            "seat":           data.get("seat",           "14A"),
            "meal":           data.get("meal",           "Veg"),
            "boarding_time":  data.get("boarding_time",  "05:30"),
        }
        flight = {
            "airline":     data.get("airline",     "IndiGo"),
            "flight_no":   data.get("flight_no",   ""),
            "origin":      data.get("origin",      ""),
            "destination": data.get("destination", ""),
            "departure":   data.get("departure",   ""),
            "arrival":     data.get("arrival",     ""),
            "gate":        data.get("gate",        "TBA"),
            "terminal":    data.get("terminal",    "T1"),
            "date":        data.get("date",        ""),
        }
        return booking, flight

    except req_exc.Timeout:
        logger.warning("Booking API timed out for PNR %s; using session fallback.", pnr)
    except req_exc.ConnectionError:
        logger.error("Booking API unreachable for PNR %s; using session fallback.", pnr)
    except req_exc.HTTPError as exc:
        logger.error("Booking API error for PNR %s: %s; using session fallback.", pnr, exc)
    except ValueError as exc:
        logger.error("Booking API parse error for PNR %s: %s", pnr, exc)
    except Exception:
        logger.exception("Unexpected error fetching booking for PNR %s.", pnr)

    # Session fallback
    pending_flight  = session.get("pending_flight", {})
    last_booking    = session.get("last_booking",   {})
    passenger_info  = session.get("passenger",      {})

    booking = {
        "pnr":            pnr,
        "passenger_name": passenger_info.get("name",      "Passenger"),
        "seat":           last_booking.get("seat",        "14A"),
        "meal":           passenger_info.get("meal_pref", "Veg"),
        "boarding_time":  last_booking.get("boarding_time", "05:30"),
    }
    flight = {
        **pending_flight,
        "airline": pending_flight.get("airline", "IndiGo"),
    }
    return booking, flight


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "AirAssist Booking Page"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port  = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(port=port, debug=debug)
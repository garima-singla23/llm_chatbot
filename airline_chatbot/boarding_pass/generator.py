# boarding_pass/generator.py

import os, io, qrcode
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

def generate_boarding_pass(booking: dict, flight: dict) -> bytes:
    """
    Generate a PDF boarding pass.
    Returns bytes of the PDF.
    """
    buffer = io.BytesIO()
    
    pnr        = booking.get("pnr", "PNRXXXXXX")
    name       = booking.get("passenger_name", "PASSENGER NAME")
    seat       = booking.get("seat", "14A")
    meal       = booking.get("meal", "Veg")
    airline    = flight.get("airline", "IndiGo")
    flight_no  = flight.get("flight_no", "6E204")
    origin     = flight.get("origin", "DEL")
    dest       = flight.get("destination", "BOM")
    departure  = flight.get("departure", "06:00")
    arrival    = flight.get("arrival", "07:31")
    gate       = flight.get("gate", "B14")
    terminal   = flight.get("terminal", "T2")
    date       = flight.get("date", datetime.now().strftime("%Y-%m-%d"))
    
    # Try to compute boarding time
    try:
        h = int(departure.split(":")[0])
        boarding = f"{max(h-1, 0):02d}:30"
    except:
        boarding = "05:00"
    
    # Page setup — boarding pass size (3.5in x 8in)
    W, H = 250*mm, 100*mm
    c = canvas.Canvas(buffer, pagesize=(W, H))
    
    # ── Background ──────────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    
    # ── Left panel (blue) ────────────────────────────────────────────────
    c.setFillColor(colors.HexColor("#1e40af"))
    c.rect(0, 0, 70*mm, H, fill=1, stroke=0)
    
    # Airline name
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(5*mm, H - 15*mm, airline.upper())
    
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#93c5fd"))
    c.drawString(5*mm, H - 22*mm, "BOARDING PASS")
    
    # Flight number
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(5*mm, H - 40*mm, flight_no)
    
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#93c5fd"))
    c.drawString(5*mm, H - 47*mm, "FLIGHT")
    
    # Date
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(5*mm, H - 60*mm, date)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#93c5fd"))
    c.drawString(5*mm, H - 67*mm, "DATE")
    
    # Class
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(5*mm, H - 80*mm, "ECONOMY")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#93c5fd"))
    c.drawString(5*mm, H - 87*mm, "CLASS")
    
    # ── Route section ────────────────────────────────────────────────────
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 32)
    c.drawString(80*mm, H - 30*mm, origin[:3].upper())
    
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.HexColor("#60a5fa"))
    c.drawCentredString(115*mm, H - 22*mm, "──── ✈ ────")
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 32)
    c.drawString(130*mm, H - 30*mm, dest[:3].upper())
    
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(80*mm, H - 35*mm, "FROM")
    c.drawString(130*mm, H - 35*mm, "TO")
    
    # Departure / Arrival
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(80*mm, H - 50*mm, departure)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(80*mm, H - 56*mm, "DEPARTS")
    
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(130*mm, H - 50*mm, arrival)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(130*mm, H - 56*mm, "ARRIVES")
    
    # ── Passenger info ───────────────────────────────────────────────────
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(80*mm, H - 68*mm, name.upper()[:25])
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawString(80*mm, H - 74*mm, "PASSENGER NAME")
    
    # Gate / Terminal / Boarding / Seat
    fields = [
        (gate, "GATE", 80),
        (terminal, "TERMINAL", 100),
        (boarding, "BOARDS AT", 120),
        (seat, "SEAT", 145),
    ]
    for val, label, x in fields:
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x*mm, H - 85*mm, str(val))
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(x*mm, H - 90*mm, label)
    
    # ── Dashed separator ─────────────────────────────────────────────────
    c.setStrokeColor(colors.HexColor("#334155"))
    c.setDash([3, 3])
    c.line(170*mm, 5*mm, 170*mm, H - 5*mm)
    c.setDash([])
    
    # ── QR code ──────────────────────────────────────────────────────────
    qr_data = f"PNR:{pnr}|FLT:{flight_no}|PAX:{name}|SEAT:{seat}"
    qr_img = qrcode.make(qr_data)
    qr_buffer = io.BytesIO()
    qr_img.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    
    c.drawImage(ImageReader(qr_buffer),
                175*mm, H//2 - 20*mm,
                width=35*mm, height=35*mm)
    
    # PNR below QR
    c.setFillColor(colors.HexColor("#60a5fa"))
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(192*mm, H//2 - 25*mm, pnr)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.drawCentredString(192*mm, H//2 - 30*mm, "BOOKING REF")
    
    c.setFillColor(colors.HexColor("#94a3b8"))
    c.setFont("Helvetica", 7)
    c.drawCentredString(192*mm, 8*mm, "AirAssist · AI Travel Concierge")
    
    c.save()
    return buffer.getvalue()


def save_boarding_pass(booking: dict, flight: dict,
                        output_dir: str = "data/boarding_passes") -> str:
    """Save boarding pass PDF to disk. Returns file path."""
    os.makedirs(output_dir, exist_ok=True)
    pnr = booking.get("pnr", "UNKNOWN")
    pdf_bytes = generate_boarding_pass(booking, flight)
    path = os.path.join(output_dir, f"{pnr}_boarding_pass.pdf")
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    print(f"[BOARDING PASS] Saved: {path}")
    return path
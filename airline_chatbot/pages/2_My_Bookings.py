# pages/2_My_Bookings.py

import streamlit as st
import sqlite3, os, io
from boarding_pass.generator import generate_boarding_pass

st.set_page_config(page_title="My Bookings", page_icon="🎫", layout="wide")
st.title("🎫 My Bookings")

email = st.text_input("Enter your email to view bookings",
                       placeholder="you@example.com")

if email:
    db = "data/bookings.db"
    if not os.path.exists(db):
        st.warning("No bookings database found.")
    else:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT * FROM bookings 
            WHERE passenger_email = ? 
            ORDER BY booked_at DESC
        """, (email,)).fetchall()
        conn.close()
        
        if not rows:
            st.info(f"No bookings found for {email}")
        else:
            st.success(f"Found {len(rows)} booking(s)")
            
            for row in rows:
                b = dict(row)
                status_color = "🟢" if b["status"] == "confirmed" else "🔴"
                
                with st.expander(
                    f"{status_color} {b['pnr']} — {b['airline']} {b['flight_no']} "
                    f"| {b['origin']} → {b['destination']} | {b['departure']}",
                    expanded=True
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Passenger**")
                        st.write(b["passenger_name"])
                        st.markdown("**Flight**")
                        st.write(f"{b['flight_no']} ({b['airline']})")
                        st.markdown("**Route**")
                        st.write(f"{b['origin']} → {b['destination']}")
                    
                    with col2:
                        st.markdown("**Date**")
                        st.write(b.get("date", "—"))
                        st.markdown("**Departure / Arrival**")
                        st.write(f"{b['departure']} → {b['arrival']}")
                        st.markdown("**Gate / Terminal**")
                        st.write(f"{b.get('gate','TBA')} · {b.get('terminal','T1')}")
                    
                    with col3:
                        st.markdown("**Seat**")
                        st.write(b["seat"])
                        st.markdown("**Meal**")
                        st.write(b.get("meal", "Veg"))
                        st.markdown("**Amount Paid**")
                        st.write(f"₹{b['amount']:,}")
                    
                    # Download boarding pass
                    booking_dict = {
                        "pnr": b["pnr"],
                        "passenger_name": b["passenger_name"],
                        "seat": b["seat"],
                        "meal": b.get("meal","Veg"),
                        "boarding_time": b.get("boarding_time","05:30"),
                    }
                    flight_dict = {
                        "airline": b["airline"],
                        "flight_no": b["flight_no"],
                        "origin": b["origin"],
                        "destination": b["destination"],
                        "departure": b["departure"],
                        "arrival": b["arrival"],
                        "gate": b.get("gate","TBA"),
                        "terminal": b.get("terminal","T1"),
                        "date": b.get("date",""),
                    }
                    
                    try:
                        pdf_bytes = generate_boarding_pass(booking_dict, flight_dict)
                        st.download_button(
                            label="📥 Download Boarding Pass (PDF)",
                            data=pdf_bytes,
                            file_name=f"{b['pnr']}_boarding_pass.pdf",
                            mime="application/pdf",
                            key=f"bp_{b['pnr']}"
                        )
                    except Exception as e:
                        st.warning(f"Boarding pass generation failed: {e}")
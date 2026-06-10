# proactive/delay_predictor.py

import random
from datetime import datetime

AIRLINE_RELIABILITY = {
    "IndiGo": 0.82,
    "Air India": 0.74,
    "SpiceJet": 0.68,
    "Vistara": 0.85,
    "Akasa Air": 0.79,
}

ROUTE_CONGESTION = {
    ("Delhi", "Mumbai"): 0.75,
    ("Delhi", "Bangalore"): 0.80,
    ("Mumbai", "Bangalore"): 0.78,
    ("Delhi", "Chennai"): 0.82,
    ("Mumbai", "Hyderabad"): 0.76,
}

PEAK_HOURS = [6, 7, 8, 9, 17, 18, 19, 20]

def get_weather_risk(destination: str) -> dict:
    random.seed(hash(f"{destination}{datetime.now().date()}") % 10000)
    score = random.uniform(0.1, 0.4)
    if score < 0.15:
        level, desc = "Low", "Clear skies expected"
    elif score < 0.25:
        level, desc = "Moderate", "Some cloud cover, minor turbulence possible"
    else:
        level, desc = "High", "Adverse weather conditions possible"
    return {"score": round(score, 2), "level": level, "description": desc}

def predict_delay(airline: str, origin: str, destination: str,
                  departure_time: str) -> dict:
    reliability = AIRLINE_RELIABILITY.get(airline, 0.75)
    
    route_key = (origin, destination)
    route_score = ROUTE_CONGESTION.get(
        route_key, ROUTE_CONGESTION.get((destination, origin), 0.75))
    
    try:
        hour = int(departure_time.split(":")[0])
        peak_factor = 0.85 if hour in PEAK_HOURS else 0.95
    except:
        peak_factor = 0.90
    
    weather = get_weather_risk(destination)
    weather_factor = 1.0 - (weather["score"] * 0.3)
    
    on_time_prob = reliability * route_score * peak_factor * weather_factor
    on_time_prob = max(0.30, min(0.95, on_time_prob))
    delay_prob = 1.0 - on_time_prob
    
    if delay_prob < 0.20:
        risk_level = "Low"
        risk_color = "green"
        recommendation = "Flight likely on time. Arrive at airport as planned."
    elif delay_prob < 0.40:
        risk_level = "Moderate"
        risk_color = "orange"
        recommendation = "Some delay risk. Monitor your flight status 2hrs before departure."
    else:
        risk_level = "High"
        risk_color = "red"
        recommendation = "High delay risk. Arrive early and check for alternatives."
    
    congestion_score = 1.0 - route_score
    if congestion_score < 0.2:
        congestion_desc = "Low — route typically clear"
    elif congestion_score < 0.35:
        congestion_desc = "Moderate — busy corridor"
    else:
        congestion_desc = "High — heavily congested route"
    
    return {
        "airline": airline,
        "route": f"{origin} → {destination}",
        "departure": departure_time,
        "on_time_probability": round(on_time_prob * 100),
        "delay_probability": round(delay_prob * 100),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "factors": {
            "airline_reliability": f"{round(reliability*100)}% historical on-time",
            "route_congestion": congestion_desc,
            "peak_hour": "Yes — expect higher delays" if hour in PEAK_HOURS else "No — off-peak",
            "weather": f"{weather['level']} risk — {weather['description']}",
        }
    }
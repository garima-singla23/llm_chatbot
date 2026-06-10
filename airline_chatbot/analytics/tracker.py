# analytics/tracker.py

import sqlite3, os, time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

ANALYTICS_DB = "data/analytics.db"

def init_analytics_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS query_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        intent TEXT,
        tools_called TEXT,
        response_time_ms INTEGER,
        was_agentic INTEGER,
        airline_filter TEXT,
        timestamp TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tool_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        tool_name TEXT,
        args TEXT,
        success INTEGER,
        duration_ms INTEGER,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()

init_analytics_db()

def log_query(session_id, user_message, intent, tools_called,
              response_time_ms, was_agentic, airline_filter=None):
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("""INSERT INTO query_log 
        (session_id, user_message, intent, tools_called,
         response_time_ms, was_agentic, airline_filter, timestamp)
        VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, user_message[:200], intent,
         ",".join(tools_called) if tools_called else "",
         response_time_ms, int(was_agentic),
         airline_filter, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def log_tool_call(session_id, tool_name, args, success, duration_ms):
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute("""INSERT INTO tool_log
        (session_id, tool_name, args, success, duration_ms, timestamp)
        VALUES (?,?,?,?,?,?)""",
        (session_id, tool_name, str(args)[:200],
         int(success), duration_ms, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_stats(hours=24):
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.row_factory = sqlite3.Row
    
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    agentic = conn.execute(
        "SELECT COUNT(*) FROM query_log WHERE was_agentic=1").fetchone()[0]
    avg_rt = conn.execute(
        "SELECT AVG(response_time_ms) FROM query_log").fetchone()[0] or 0
    
    intents = conn.execute("""
        SELECT intent, COUNT(*) as cnt FROM query_log 
        WHERE intent != '' GROUP BY intent ORDER BY cnt DESC
    """).fetchall()
    
    tools = conn.execute("""
        SELECT tool_name, COUNT(*) as cnt FROM tool_log 
        GROUP BY tool_name ORDER BY cnt DESC
    """).fetchall()
    
    recent = conn.execute("""
        SELECT user_message, intent, response_time_ms, timestamp 
        FROM query_log ORDER BY id DESC LIMIT 10
    """).fetchall()
    
    conn.close()
    return {
        "total_queries": total,
        "agentic_queries": agentic,
        "faq_queries": total - agentic,
        "avg_response_ms": round(avg_rt),
        "agentic_pct": round(agentic/max(total,1)*100),
        "intent_distribution": [dict(r) for r in intents],
        "tool_usage": [dict(r) for r in tools],
        "recent_queries": [dict(r) for r in recent],
    }
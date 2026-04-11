import pandas as pd
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_CSV_PATH = BASE_DIR / "all_chunks.csv"

_EVAL_SPECS = [
    ("How many carry-on bags are allowed?", "one (1) carry-on bag"),
    ("Where must personal items be stored?", "stored in the overhead bin"),
    ("What is the fee for first checked bag on domestic flights?", "Domestic (US/Canada) $35 USD"),
    ("Are spare lithium batteries allowed in checked luggage?", "Spare Batteries: Strictly PROHIBITED"),
    ("Can I carry a violin as cabin baggage?", "Violins and trumpets may be carried"),
    ("What happens if I cancel within 24 hours?", "100% refund"),
    ("How long does a credit card refund take?", "Credit Card: 7–10 business days"),
    ("What qualifies as a significant schedule change?", "delay of more than 3 hours"),
    ("Is there a waiver for medical emergencies?", "Medical Emergencies"),
    ("What happens if I miss my flight without canceling?", "ticket is considered"),
    ("When does online check-in open?", "Opening Window: 48 hours"),
    ("What is the bag drop deadline for domestic flights?", "Domestic 2 Hours Before"),
    ("What documents are required for domestic travel?", "REAL ID Compliance"),
    ("Is ETIAS required for EU travel?", "ETIAS authorization"),
    ("Can unaccompanied minors check in online?", "Online check-in is NOT available"),
]


def find_chunk_id_by_phrase(phrase, df):

    for _, row in df.iterrows():
        if phrase.lower() in row["text"].lower():
            return row["chunk_id"]

    raise ValueError(f"Phrase not found: {phrase}")


@lru_cache(maxsize=1)
def get_evaluation_data():
    df = pd.read_csv(CHUNKS_CSV_PATH)

    return [
        {
            "query": query,
            "ground_truth_doc": find_chunk_id_by_phrase(phrase, df)
        }
        for query, phrase in _EVAL_SPECS
    ]

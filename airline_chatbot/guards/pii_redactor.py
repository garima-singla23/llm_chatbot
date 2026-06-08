import re
from typing import List, Tuple

try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    _PRESIDIO_AVAILABLE = True
except Exception:
    AnalyzerEngine = None
    Pattern = None
    PatternRecognizer = None
    AnonymizerEngine = None
    _PRESIDIO_AVAILABLE = False


# Indian PII patterns
Aadhaar = r"\b[2-9]\d{11}\b"
IndianPhone = r"\b[6-9]\d{9}\b"
PAN = r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"
CreditCard = r"\b(?:\d[ -]*?){13,19}\b"


def _regex_redact(text: str) -> Tuple[str, List[str]]:
    findings: List[str] = []

    patterns = [
        ("AADHAAR", Aadhaar),
        ("PHONE", IndianPhone),
        ("PAN", PAN),
        ("CARD", CreditCard),
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ]

    redacted = text

    for label, pattern in patterns:
        matches = list(re.finditer(pattern, redacted))
        if not matches:
            continue

        findings.append(label)
        redacted = re.sub(pattern, f"<{label}>", redacted)

    return redacted, findings


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """
    Redact common PII from text.
    Falls back to regex-only mode if Presidio is unavailable.
    """

    if not _PRESIDIO_AVAILABLE:
        return _regex_redact(text)

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    custom_patterns = [
        Pattern(name="aadhaar", regex=Aadhaar, score=0.85),
        Pattern(name="phone", regex=IndianPhone, score=0.85),
        Pattern(name="pan", regex=PAN, score=0.85),
        Pattern(name="card", regex=CreditCard, score=0.85),
    ]

    recognizer = PatternRecognizer(
        supported_entity="CUSTOM_PII",
        patterns=custom_patterns,
    )

    analyzer.registry.add_recognizer(recognizer)

    results = analyzer.analyze(
        text=text,
        language="en",
        entities=["CUSTOM_PII", "EMAIL_ADDRESS"],
    )

    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
    ).text

    findings = sorted({res.entity_type for res in results})

    return redacted, findings
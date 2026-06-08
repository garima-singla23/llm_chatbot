from guards.pii_redactor import redact_pii

test = "My Aadhaar is 234567890123, card 4111111111111111, phone 9876543210"

clean, _ = redact_pii(test)

print("Original:", test)
print("Redacted:", clean)
print(
    "Pass:",
    "234567890123" not in clean
    and "4111111111111111" not in clean
)
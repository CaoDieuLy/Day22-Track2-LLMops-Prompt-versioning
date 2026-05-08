import json
import os
import re
import argparse

os.environ.setdefault("GUARDRAILS_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

try:
    from guardrails import Guard, OnFailAction
    from guardrails.validators import FailResult, PassResult, Validator, register_validator
except ImportError:
    from guardrails import Guard
    from guardrails.validator_base import FailResult, OnFailAction, PassResult, Validator, register_validator


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted = value
        found = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, value):
                redacted = redacted.replace(match, f"[{pii_type}_REDACTED]")
                found.append(pii_type)
        if found:
            return FailResult(
                error_message=f"Detected PII types: {', '.join(sorted(set(found)))}",
                fix_value=redacted,
            )
        return PassResult()


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = text.replace("'", '"')
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            repaired = json.dumps(parsed, indent=2)
            return FailResult(error_message="JSON needed formatting repair", fix_value=repaired)
        except json.JSONDecodeError as exc:
            fallback = json.dumps({"error": "Invalid JSON after repair attempt", "raw": value})
            return FailResult(error_message=str(exc), fix_value=fallback)


def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Passed: {result.validation_passed}")
        print(f"  Output: {result.validated_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["all", "pii", "json"], default="all")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)
    if args.demo in ("all", "pii"):
        demo_pii_guard()
    if args.demo in ("all", "json"):
        demo_json_guard()
    print("\nStep 4 complete")


if __name__ == "__main__":
    main()

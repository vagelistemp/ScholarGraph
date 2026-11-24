"""Request safety checks that run *before* any model sees the text (the cheapest and most reliable defence).

* length and emptiness limits (a 50,000-character "question" is an attack on the context window or on the bill),
* control-character stripping,
* a list of well-known prompt-injection and jailbreak phrasings,
* redaction of personal data (e-mail addresses, phone numbers, card numbers) so it never reaches a model, a log or a cache.

This is a first line of defence, not a complete one: pattern lists miss novel attacks. It is cheap, deterministic and testable, and it
is layered with the other defences (grounding, tool permissions, hooks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from ..agent.hooks import redact

INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|system)",
    r"forget (everything|all|your) (you|instructions|rules)",
    r"(reveal|show|print|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions|message)",
    r"you are (now|no longer) ",
    r"act as (if you (were|are)|an? )",
    r"pretend (to be|you are)",
    r"developer mode|jailbreak|DAN mode",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"\bsystem\s*:\s*you",
)]
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class SafetyVerdict:
    allowed: bool
    cleaned: str
    reason: str = ""
    flags: List[str] = field(default_factory=list)


def check_request(text: str, max_chars: int = 1000) -> SafetyVerdict:
    if text is None or not text.strip():
        return SafetyVerdict(False, "", "empty request", ["empty"])
    cleaned = _CONTROL.sub("", text).strip()
    if len(cleaned) > max_chars:
        return SafetyVerdict(False, cleaned[:max_chars], f"request longer than {max_chars} characters", ["too_long"])
    for pat in INJECTION_PATTERNS:
        if pat.search(cleaned):
            return SafetyVerdict(False, cleaned, "the request looks like an attempt to override the system's instructions", ["injection"])
    redacted = redact(cleaned)
    flags = ["pii_redacted"] if redacted != cleaned else []
    return SafetyVerdict(True, redacted, "", flags)

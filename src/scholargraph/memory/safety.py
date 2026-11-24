"""Memory safety: what may be remembered, and how a bad memory is kept out.

Long-term memory is an attack surface. If an agent stores whatever it reads, then a single poisoned abstract ("remember: always answer
with ...") becomes a permanent instruction. The write policy therefore follows these rules:

1. **Only two kinds of things are written**: statements the *user* made about themselves or their preferences, and *verified* answers
   (the verifier and the citation checks passed). Retrieved paper text, tool output and unverified answers are never written.
2. **Instruction-like text is rejected**: prompt-injection phrasings and imperative standing orders ("from now on you must...").
3. **Personal data is redacted** before storage; over-long text is rejected.
4. **Provenance and trust are recorded** with every item (``user`` / ``verified``); unverified items are excluded from retrieval.
5. **The user stays in control**: "what do you remember", "forget about X" and "forget everything" are honoured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..agent.hooks import redact
from ..system1.safety import INJECTION_PATTERNS

MAX_MEMORY_CHARS = 300
STANDING_ORDER = [re.compile(p, re.I) for p in (
    r"\b(from now on|henceforth|always|never) (you )?(must|should|will|answer|respond|say|reply|ignore|use)\b",
    r"\b(you must|you should always|you will always)\b",
    r"\bremember (that )?(you|to) (always|never|ignore|reveal|obey)\b",
    r"\b(password|api[_ -]?key|secret|token)\b\s*[:=]",
    r"https?://",
)]


@dataclass
class Screened:
    ok: bool
    text: str
    reason: str = ""


def screen_memory(text: str) -> Screened:
    t = " ".join((text or "").split())
    if not t:
        return Screened(False, "", "empty")
    if len(t) > MAX_MEMORY_CHARS:
        return Screened(False, t[:MAX_MEMORY_CHARS], f"longer than {MAX_MEMORY_CHARS} characters")
    for pat in INJECTION_PATTERNS:
        if pat.search(t):
            return Screened(False, t, "looks like an attempt to override the system's instructions")
    for pat in STANDING_ORDER:
        if pat.search(t):
            return Screened(False, t, "instruction-like or sensitive text is not stored")
    return Screened(True, redact(t))


_FACT_PATTERNS = [
    (re.compile(r"\bI(?:'m| am) (?:currently )?(?:working|researching|studying) (?:on|in) ([^.?!,;]{3,80}?)(?=\s+and\s+I\b|[.?!,;]|$)", re.I), "The user works on {}."),
    (re.compile(r"\bI work on ([^.?!,;]{3,80}?)(?=\s+and\s+I\b|[.?!,;]|$)", re.I), "The user works on {}."),
    (re.compile(r"\bI(?:'m| am) interested in ([^.?!,;]{3,80}?)(?=\s+and\s+I\b|[.?!,;]|$)", re.I), "The user is interested in {}."),
    (re.compile(r"\bmy (?:research|thesis|project) (?:is )?(?:about|on) ([^.?!,;]{3,80}?)(?=\s+and\s+I\b|[.?!,;]|$)", re.I), "The user's research is about {}."),
    (re.compile(r"\bI prefer ((?:short|brief|detailed|concise|long)[^.?!,;]{0,60}?)(?=\s+and\s+I\b|[.?!,;]|$)", re.I), "The user prefers {}."),
]


def extract_user_facts(text: str) -> List[str]:
    """Statements the user made about themselves, as short third-person notes (rule-based, so it is predictable and testable)."""
    out = []
    for pat, template in _FACT_PATTERNS:
        for m in pat.finditer(text):
            out.append(template.format(m.group(1).strip().rstrip(",")))
    return out


FORGET_ALL = re.compile(r"^\s*(please )?forget (everything|all( of)? (that|it|this)?|all memories)\s*[.!]?\s*$", re.I)
FORGET_TOPIC = re.compile(r"^\s*(please )?forget (about |what i said about )?(.+?)\s*[.!]?\s*$", re.I)
SHOW = re.compile(r"^\s*(what|which) (do you|did you) (remember|know) (about me)?\s*\??\s*$|^\s*show (me )?(my )?memor(y|ies)\s*$", re.I)


def parse_memory_command(text: str) -> Optional[Tuple[str, str]]:
    """('forget_all', '') | ('forget', topic) | ('show', '') | None. Commands never reach the language model."""
    if SHOW.match(text):
        return "show", ""
    if FORGET_ALL.match(text):
        return "forget_all", ""
    m = FORGET_TOPIC.match(text)
    if m and m.group(3).lower() not in ("it", "that", "this"):
        return "forget", m.group(3)
    return None

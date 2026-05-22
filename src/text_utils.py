import re
from typing import Optional


def normalize_compare_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = text.replace("ё", "е")
    # Remove non-alphanumeric characters, keeping Cyrillic and Latin
    text = re.sub(r"[^a-zа-я0-9\s]", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_too_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    na = normalize_compare_text(a)
    nb = normalize_compare_text(b)

    if not na or not nb:
        return False

    if na == nb:
        return True

    if na in nb or nb in na:
        shorter = min(len(na), len(nb))
        longer = max(len(na), len(nb))
        if longer > 0 and (shorter / longer) >= threshold:
            return True

    words_a = set(na.split())
    words_b = set(nb.split())

    if words_a and words_b:
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        if union > 0 and (overlap / union) >= threshold:
            return True

    return False


def is_short_text(text: str) -> bool:
    return len((text or "").strip()) < 12


def strip_speaker_prefix(text: str) -> str:
    if not text:
        return ""
    return re.sub(
        r"^[^:]{1,32}:\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )


def strip_output_labels(text: str) -> str:
    if not text:
        return ""

    prefixes = (
        "assistant:",
        "nika:",
        "ника:",
        "bot:",
        "assistant response:",
        "response:",
        "reply:",
    )

    out = text.strip()

    changed = True
    while changed:
        changed = False
        low = out.lower()
        for p in prefixes:
            if low.startswith(p):
                out = out[len(p):].strip()
                changed = True
                break

    return out


def clean_response(text: str) -> Optional[str]:
    if not text:
        return None

    text = strip_output_labels(text)
    text = strip_speaker_prefix(text)
    text = text.strip()

    return text or None


def sanitize_summary_text(text: str) -> str:
    if not text:
        return ""

    text = strip_output_labels(text)

    bad_prefixes = [
        "короткая сводка:",
        "сводка:",
        "summary:",
        "short summary:",
        "итог:",
    ]

    low = text.lower()
    for prefix in bad_prefixes:
        if low.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 2:
            lines.append(line)

    return "\n".join(lines).strip()


def extract_json_object(text: str) -> dict:
    if not text:
        return {}

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    import json

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def extract_channel_tokens(text: str) -> list[str]:
    return re.findall(r"<#(\d+)>", text or "")


def extract_user_tokens(text: str) -> list[str]:
    return re.findall(r"<@!?(\d+)>", text or "")


def attachment_summary(atts) -> str:
    if not atts:
        return ""

    parts = []
    for a in atts:
        name = getattr(a, "filename", "file")
        parts.append(name)

    return ", ".join(parts)

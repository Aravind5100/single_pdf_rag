"""A deterministic grounding check: every number in an answer must appear in the retrieved context.

The LLM verifier is one small yes/no call and misses invented details (see eval/COMPARISON.md: it
accepted 8 of 12 answers that had one invented sentence appended). A made-up dollar amount, count or
time is the commonest such detail in policy documents, and a string comparison catches it exactly.

What it does NOT do: it cannot catch an invented fact that contains no number ("a student panel
decides"), a wrong claim that reuses a real number, or an omission. It is a cheap extra guard next to
the verifier, not a replacement.
"""
import re
from typing import List

# Small number words an answer might write instead of digits. "one" is left out on purpose: it is
# usually the pronoun or "one of the following", not a quantity.
_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100",
}
_CITATION = re.compile(r"[\[(]\s*pages?\s*\d+(?:\s*[,&-]\s*\d+)*\s*[\])]|\bpages?\s+\d+\b", re.I)
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")        # "1. " or "2) " at the start of a line
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_WORD = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.I)
_DIGITS = re.compile(r"\d+")


def numbers_in(text: str, answer_side: bool = False) -> List[str]:
    """Digit runs in `text`, after turning number words into digits and removing thousands commas.

    For an answer (`answer_side=True`) page citations and numbered-list markers are ignored first,
    because those digits are formatting, not claims.
    """
    if answer_side:
        text = _CITATION.sub(" ", text)
        text = _LIST_MARKER.sub("", text)
    text = _THOUSANDS.sub("", text)
    text = _WORD.sub(lambda m: f" {_NUMBER_WORDS[m.group(1).lower()]} ", text)
    return _DIGITS.findall(text)


def ungrounded_numbers(answer: str, context: str) -> List[str]:
    """Numbers that appear in the answer but nowhere in the context (each reported once, in order)."""
    context_numbers = set(numbers_in(context))
    seen, missing = set(), []
    for n in numbers_in(answer, answer_side=True):
        if n not in context_numbers and n not in seen:
            seen.add(n)
            missing.append(n)
    return missing

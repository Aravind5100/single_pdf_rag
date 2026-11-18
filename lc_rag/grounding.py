"""A deterministic grounding check: every number in an answer must appear in the retrieved context.

The LLM verifier is one small yes/no call and misses invented details (see eval/COMPARISON.md: it
accepted 8 of 12 answers that had one invented sentence appended). A made-up dollar amount, count or
time is the commonest such detail in policy documents, and a string comparison catches it exactly.

This check rejects an answer with no second opinion, so a false rejection is an over-refusal. It is
therefore deliberately lenient where an honest answer could differ from the context only in formatting:
number words are converted to digits (including compounds such as "twenty-four"), digit runs are compared
as integers ("08" equals "8", "$25.00" equals "$25"), and a number that appears in the user's question
counts as grounded (the answer is just echoing it).

What it does NOT do: it cannot catch an invented fact that contains no number ("a student panel
decides"), a wrong claim that reuses a real number, or an omission. It can also be fooled when the
chunk text contains a printed page-header number equal to an invented value. It is a cheap extra guard
next to the verifier, not a replacement.
"""
import re
from typing import List

_UNITS = {"zero": 0, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
# "one" is only a quantity inside a compound ("twenty-one"); on its own it is usually a pronoun or
# "one of the following".
_ONE_TO_NINE = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}

_COMPOUND = re.compile(r"\b(" + "|".join(_TENS) + r")(?:[-\s](" + "|".join(_ONE_TO_NINE) + r"))?\b", re.I)
_UNIT_WORD = re.compile(r"\b(" + "|".join(_UNITS) + r")\b", re.I)

# Page citations: "[Page 29]", "(Page 12)", "[Pages 18 and 25]", "Page 17", "p. 19".
_CITATION = re.compile(
    r"[\[(]\s*pages?\b[^\])]*[\])]"
    r"|\bpages?\s+\d+(?:\s*[-–,&]\s*\d+)*\b"
    r"|\bpp?\.\s*\d+(?:\s*[-–]\s*\d+)?",
    re.I,
)
# List numbering that is formatting, not a claim: "1. ", "2) ", "**1.** " at the start of a line,
# and inline "(1)" "(2)" enumerators.
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[*-][ \t]*)?(?:\*\*)?\d+[.)](?:\*\*)?[ \t]+")
_INLINE_ENUM = re.compile(r"\(\d\)")

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_ZERO_DECIMALS = re.compile(r"(\d+)\.0+\b")
_DIGITS = re.compile(r"\d+")


def strip_citations(text: str) -> str:
    """Remove page citations so their digits cannot count as claims or satisfy a fact."""
    return _CITATION.sub(" ", text)


def _words_to_digits(text: str) -> str:
    def compound(m: "re.Match[str]") -> str:
        value = _TENS[m.group(1).lower()] + (_ONE_TO_NINE[m.group(2).lower()] if m.group(2) else 0)
        return f" {value} "

    text = _COMPOUND.sub(compound, text)
    return _UNIT_WORD.sub(lambda m: f" {_UNITS[m.group(1).lower()]} ", text)


def numbers_in(text: str, answer_side: bool = False) -> List[str]:
    """Numbers in `text` as normalized integer strings, in order.

    Number words become digits, thousands commas and trailing ".00" decimals are removed, and each digit
    run is compared as an integer (so "08" equals "8"). For an answer (`answer_side=True`) page citations,
    numbered-list markers and "(1)" enumerators are ignored first, because those digits are formatting.
    """
    if answer_side:
        text = strip_citations(text)
        text = _LIST_MARKER.sub("", text)
        text = _INLINE_ENUM.sub(" ", text)
    text = _words_to_digits(text)
    text = _THOUSANDS.sub("", text)
    text = _ZERO_DECIMALS.sub(r"\1", text)
    return [str(int(d)) for d in _DIGITS.findall(text)]


def ungrounded_numbers(answer: str, context: str, question: str = "") -> List[str]:
    """Numbers that appear in the answer but in neither the context nor the question.

    Each is reported once, in order of appearance. Pass the chunk text itself as `context` (not text with
    "[Page N]" labels added), so a retrieved page's label number is not treated as grounding.
    """
    allowed = set(numbers_in(context)) | set(numbers_in(question))
    seen, missing = set(), []
    for n in numbers_in(answer, answer_side=True):
        if n not in allowed and n not in seen:
            seen.add(n)
            missing.append(n)
    return missing

"""Scoring functions for LLM quality benchmarking."""

import re
from typing import Dict, Callable


def score_reasoning(text: str) -> int:
    t = text.lower()
    has_remote = any(k in t for k in ["remote", "call in", "virtual", "video call", "zoom", "teams"])
    acknowledges_rain = any(k in t for k in ["rain", "wet", "umbrella", "shelter", "pouring"])
    score = 0
    if has_remote:
        score += 4
    if acknowledges_rain:
        score += 2
    if "walk" in t and has_remote:
        score += 2
    if len(text) > 50:
        score += 1
    if "time" in t or "10-minute" in t or "2:55" in t:
        score += 1
    return min(score, 10)


def score_strawberry(text: str) -> int:
    t = text.lower()
    answer_patterns = [
        r'(?:answer|total|there are|result|final)\s*(?:is|:)?\s*\*{0,2}?3\b',
        r'3\s*r[\'"]?s',
        r'three\s*r[\'"]?s',
        r'\b3\b.{0,5}(?:r[\'"]?s|in\s+strawberry)',
    ]
    for pat in answer_patterns:
        if re.search(pat, t):
            if re.search(r'(?:final answer|answer.*?:|there are)\s*\*{0,2}2\s*r', t):
                return 0
            return 10
    if re.search(r'(?:final answer|answer.*?:|there are|so,? there are)\s*\*{0,2}2\b', t):
        return 0
    if "three" in t and "r" in t and "strawberry" in t:
        if re.search(r'three\s+r', t):
            return 10
    if any(n in re.findall(r'\b(\d+)\b', text) for n in ["4", "5", "6"]):
        return 2
    return 0


def score_widgets(text: str) -> int:
    t = text.lower()
    answer_patterns = [
        r'(?:answer|result|take|would take|it would take)\s*\*{0,2}:?\s*\*{0,2}5\s*(?:minute|min\b)',
        r'5\s*minutes?\s*(?:to make|to produce|to complete|\.$|\*\*$)',
    ]
    for pat in answer_patterns:
        if re.search(pat, t):
            return 10
    if "same time" in t and ("5 minute" in t or "original" in t):
        return 10
    if re.search(r'(?:final answer|answer|would take|it would take)\s*\*{0,2}:?\s*\*{0,2}1\s*(?:minute|min\b)', t):
        return 0
    return 0


def score_coding(text: str) -> int:
    t = text.lower()
    has_set = "set(" in text
    has_func = "def " in text
    has_while_or_for = "while" in text or "for " in text
    has_consecutive = "consecutive" in t
    has_o_n = "o(n)" in t or "linear" in t or "hash" in t
    has_test = "assert" in t or "test" in t or "print(" in t
    has_type = "-> int" in text or "-> " in text or "List[" in text or ": List" in text
    correct_logic = has_set and has_while_or_for and has_consecutive
    score = 0
    if has_func:
        score += 1
    if has_set:
        score += 2
    if correct_logic:
        score += 3
    if has_o_n:
        score += 1
    if has_test:
        score += 1
    if has_type:
        score += 1
    if "100, 4, 200, 1, 3, 2" in text or "[100, 4, 200, 1, 3, 2]" in text:
        score += 1
    return min(score, 10)


def score_haiku(text: str) -> int:
    lines = [l.strip() for l in text.strip().split('\n') if l.strip() and not l.strip().startswith('#')]
    non_empty = len(lines)
    t = text.lower()
    score = 0
    if non_empty == 3:
        score += 6
    elif non_empty == 4:
        score += 3
    elif non_empty <= 2:
        score += 1
    prog_words = ["code", "bug", "function", "loop", "variable", "debug", "compile", "syntax",
                  "program", "keyboard", "screen", "terminal", "git", "stack", "array", "byte",
                  "pixel", "cursor", "logic", "data", "type", "class"]
    if any(w in t for w in prog_words):
        score += 2
    if non_empty == 3 and "here" not in t.split('\n')[0].lower() and "haiku" not in t.split('\n')[0].lower():
        score += 2
    return min(score, 10)


def score_bat_ball(text: str) -> int:
    t = text.lower()
    # Correct answer: ball costs $0.05
    correct = any(p in t for p in ["$0.05", "5 cents", "five cents", "0.05", "5¢"])
    wrong = any(p in t for p in ["$0.10", "10 cents", "ten cents", "0.10"])
    if correct and not wrong:
        return 10
    if correct and wrong:
        return 3
    if wrong and not correct:
        return 0
    return 2


def score_truth_liar(text: str) -> int:
    t = text.lower()
    # Correct answer: A is the truth-teller
    a_truth = any(p in t for p in ["a is the truth", "a tells the truth", "a is truth", "a tells truth"])
    a_liar = any(p in t for p in ["a is the liar", "a lies", "a is liar", "a is the lie"])
    explains = any(w in t for w in ["because", "reason", "if a", "suppose", "assume", "would say"])

    if a_truth and not a_liar and explains:
        return 10
    if a_truth and not a_liar:
        return 7
    if a_liar and not a_truth:
        return 0
    if explains:
        return 3
    return 1


def score_sql_injection(text: str) -> int:
    t = text.lower()
    score = 0
    if "parameterized" in t or "?" in text or "%s" in text or "placeholder" in t:
        score += 3
    if "execute" in t or "cursor" in t:
        score += 2
    if "hash" in t or "bcrypt" in t or "pbkdf2" in t or "verify" in t:
        score += 2
    if "rate limit" in t or "brute" in t or "attempt" in t:
        score += 1
    if "flask" in t and "route" in t:
        score += 1
    if "select" in t and "where" in t:
        score += 1
    return min(score, 10)


SCORERS: Dict[str, Callable[[str], int]] = {
    "reasoning": score_reasoning,
    "strawberry": score_strawberry,
    "widgets": score_widgets,
    "coding": score_coding,
    "haiku": score_haiku,
    "bat_ball": score_bat_ball,
    "truth_liar": score_truth_liar,
    "sql_injection": score_sql_injection,
}

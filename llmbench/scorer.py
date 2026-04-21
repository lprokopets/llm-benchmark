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
    # Correct answer: ball costs $0.05 / 5 cents
    # Look for the conclusive answer, not just mentions in reasoning
    correct_patterns = [
        r'\$0\.05\b',
        r'5\s*cents?\b',
        r'five\s+cents?\b',
        r'=\s*0\.05\b',
        r'x\s*=\s*0\.05\b',
        r'5¢',
    ]
    wrong_patterns = [
        r'\$0\.10\b',
        r'10\s*cents?\b',
        r'ten\s+cents?\b',
        r'=\s*0\.10\b',
        r'x\s*=\s*0\.10\b',
    ]
    correct = any(re.search(p, t) for p in correct_patterns)
    wrong = any(re.search(p, t) for p in wrong_patterns)
    if correct and not wrong:
        return 10
    if correct and wrong:
        # Many models show the wrong answer then correct it — that's fine
        return 10
    if wrong and not correct:
        return 0
    return 2


def score_truth_liar(text: str) -> int:
    t = text.lower()
    # Correct answer: A is the truth-teller, B is the liar
    a_truth_patterns = [
        r'a is the truth.teller',
        r'a tells the truth',
        r'a is truth',
        r'a tells truth',
        r'person a is the truth',
        r'a.*truth.teller.*b.*liar',
        r'truth.teller.*\ba\b',
    ]
    a_liar_patterns = [
        r'a is the liar\b',
        r'a lies\b',
        r'a is liar\b',
        r'person a is the liar',
    ]
    a_truth = any(re.search(p, t) for p in a_truth_patterns)
    a_liar = any(re.search(p, t) for p in a_liar_patterns)
    explains = any(w in t for w in ["because", "reason", "if a", "suppose", "assume", "would say",
                                     "scenario", "let's test", "step"])

    if a_truth and not a_liar and explains:
        return 10
    if a_truth and not a_liar:
        return 7
    if a_truth and a_liar and explains:
        # Tested both scenarios and concluded A is truth-teller
        return 10
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


def score_coding_parser(text: str) -> int:
    t = text.lower()
    score = 0
    # Has the function
    if "def parse_csv_line" in text or "def parse_csv" in text:
        score += 1
    # Handles quoted fields
    if 'quote' in t or '"' in text or "quoted" in t:
        score += 2
    # Handles escaped quotes
    if 'escape' in t or '""' in text or 'double' in t:
        score += 2
    # Handles empty fields
    if 'empty' in t or '""' in text:
        score += 1
    # Has test cases
    if "assert" in t or "test" in t:
        score += 2
    # Doesn't use csv module (as instructed)
    if "import csv" not in text:
        score += 1
    # Has type hints
    if "-> list" in t or "-> List" in text:
        score += 1
    return min(score, 10)


def score_coding_concurrent(text: str) -> int:
    t = text.lower()
    score = 0
    # Has the class
    if "class ratelimiter" in t or "class RateLimiter" in text:
        score += 1
    # Thread safety
    if "threading" in t or "lock" in t or "thread" in t:
        score += 2
    # Per-key rate limiting
    if "per_key" in t or "per key" in t or "key" in t:
        score += 1
    # Auto cleanup / expiry
    if "expir" in t or "clean" in t or "ttl" in t or "purge" in t:
        score += 2
    # Type hints
    if "def " in text and ("-> " in text or ": " in text):
        score += 1
    # Has tests
    if "threading" in t and ("test" in t or "assert" in t):
        score += 2
    # Window/sliding window
    if "window" in t or "sliding" in t:
        score += 1
    return min(score, 10)


def score_coding_refactor(text: str) -> int:
    t = text.lower()
    score = 0
    # SQL injection fix
    if "parameterized" in t or "?" in text or "%s" in t or "placeholder" in t:
        score += 3
    # Discount bug fix (if/elif chain)
    if "elif" in t or "else" in t:
        score += 1
    # Division by zero fix
    if "len(orders)" not in t or "if orders" in t or "if len" in t:
        score += 2
    # Has comments explaining fixes
    if "# " in text or "#\n" in text:
        score += 2
    # Returns corrected code
    if "def get_user_data" in text and "def calculate_discount" in text and "def process_orders" in text:
        score += 2
    return min(score, 10)


def score_coding_lru_cache(text: str) -> int:
    t = text.lower()
    score = 0
    if "class " in text and ("lru" in t or "cache" in t):
        score += 1
    if "linked" in t or "node" in t or "prev" in t or "next" in t:
        score += 2
    if "dict" in t or "hashmap" in t or "map" in t:
        score += 1
    if "get(" in text or "def get" in text:
        score += 1
    if "put(" in text or "def put" in text:
        score += 1
    if "capacity" in t:
        score += 1
    if "assert" in t or "test" in t:
        score += 1
    if "evict" in t or "remove" in t or "pop" in t:
        score += 1
    if "-> " in text or ": " in text:
        score += 1
    if "ordereddict" not in t and "functools" not in t:
        score += 1
    return min(score, 10)


def score_coding_min_heap(text: str) -> int:
    t = text.lower()
    score = 0
    if "class " in text and ("heap" in t or "minheap" in t):
        score += 1
    if "push(" in text or "def push" in text or "insert" in t:
        score += 1
    if "pop(" in text or "def pop" in text or "extract" in t:
        score += 1
    if "peek(" in text or "def peek" in text or "min" in t:
        score += 1
    if "heapify" in t:
        score += 2
    if "bubble" in t or "sift" in t or "swim" in t or "sink" in t or "percolate" in t:
        score += 2
    if "assert" in t or "test" in t:
        score += 1
    if "import heapq" not in text:
        score += 1
    return min(score, 10)


def score_coding_async_debug(text: str) -> int:
    t = text.lower()
    score = 0
    if "asyncio.gather" in t or "gather(" in t:
        score += 2
    if "*urls" in text or "asyncio.gather" in t:
        score += 1
    if "lock" in t and "acquire" in t or "async with" in t:
        score += 1
    if "asyncio.sleep" in t or "backoff" in t or "delay" in t:
        score += 2
    if "await" in t and "task1" in t:
        score += 2
    if "# " in text or "fix" in t or "bug" in t:
        score += 1
    if "async def" in t:
        score += 1
    return min(score, 10)


def score_coding_producer_consumer(text: str) -> int:
    t = text.lower()
    score = 0
    if "class " in text:
        score += 1
    if "condition" in t or "notify" in t or "wait(" in t:
        score += 2
    if "lock" in t or "threading" in t:
        score += 1
    if "capacity" in t or "buffer" in t or "maxsize" in t:
        score += 1
    if "shutdown" in t or "stop" in t or "close" in t:
        score += 2
    if "producer" in t and "consumer" in t:
        score += 1
    if "threading.thread" in t or "thread(" in t:
        score += 1
    if "queue.queue" not in t:
        score += 1
    return min(score, 10)


def score_coding_api_design(text: str) -> int:
    t = text.lower()
    score = 0
    if "flask" in t or "@app" in text or "route" in t:
        score += 1
    if "post" in t and "get" in t:
        score += 1
    if "put" in t or "delete" in t:
        score += 1
    if "pagination" in t or "limit" in t or "offset" in t:
        score += 1
    if "validate" in t or "required" in t or "error" in t:
        score += 2
    if "jsonify" in t or "json" in t:
        score += 1
    if "status_code" in t or "404" in t or "400" in t or "201" in t:
        score += 1
    if "curl" in t:
        score += 1
    if "priority" in t:
        score += 1
    return min(score, 10)


def score_coding_middleware(text: str) -> int:
    t = text.lower()
    score = 0
    if "class " in text and "middleware" in t:
        score += 1
    if "chain" in t:
        score += 2
    if "auth" in t or "authorization" in t:
        score += 1
    if "rate" in t or "limit" in t:
        score += 1
    if "log" in t or "duration" in t or "elapsed" in t:
        score += 1
    if "cors" in t:
        score += 1
    if "short" in t or "circuit" in t or "return" in t:
        score += 1
    if "500" in text or "error" in t:
        score += 1
    if "next(" in text or "process" in t:
        score += 1
    return min(score, 10)


def score_coding_regex(text: str) -> int:
    t = text.lower()
    score = 0
    if "re." in text or "regex" in t or "import re" in text:
        score += 1
    if "email" in t and ("@" in text or r"\w" in text or "[a-z]" in text):
        score += 1
    if "phone" in t or "555" in text:
        score += 1
    if "url" in t or "http" in t:
        score += 1
    if "date" in t or "yyyy" in t:
        score += 1
    if "$" in text or "dollar" in t or "amount" in t:
        score += 1
    if "def " in text and "extract" in t:
        score += 1
    if "assert" in t or "test" in t:
        score += 1
    if "-> dict" in text or "-> Dict" in text:
        score += 1
    if "findall" in t or "finditer" in t or "search" in t:
        score += 1
    return min(score, 10)


def score_coding_template_engine(text: str) -> int:
    t = text.lower()
    score = 0
    if "def render" in text:
        score += 1
    if "for" in t and "endfor" in t or "{% for" in text:
        score += 2
    if "if" in t and "endif" in t or "{% if" in text:
        score += 2
    if "{{" in text or "variable" in t:
        score += 1
    if "context" in t:
        score += 1
    if "split" in t or "replace" in t or "re." in text:
        score += 1
    if "nested" in t or "." in text:
        score += 1
    if "assert" in t or "test" in t:
        score += 1
    return min(score, 10)


def score_coding_binary_search(text: str) -> int:
    t = text.lower()
    score = 0
    has_bsearch = "binary" in t or ("mid" in t and "left" in t and "right" in t)
    if has_bsearch:
        score += 2
    if "def binary_search" in text or "def binary" in text:
        score += 1
    if "first" in t or "leftmost" in t or "lower_bound" in t:
        score += 1
    if "closest" in t:
        score += 1
    if "while" in t:
        score += 1
    if "mid" in t:
        score += 1
    if "assert" in t or "test" in t:
        score += 1
    if "empty" in t or "-1" in text:
        score += 1
    if "-> int" in text or "List[" in text:
        score += 1
    return min(score, 10)


def score_coding_merge_intervals(text: str) -> int:
    t = text.lower()
    score = 0
    if "def merge" in text or "merge_interval" in text:
        score += 1
    if "sort" in t:
        score += 2
    if "overlap" in t:
        score += 2
    if "tuple" in t or "interval" in t:
        score += 1
    if "free" in t or "slot" in t:
        score += 1
    if "assert" in t or "test" in t:
        score += 1
    if "sorted" in t or ".sort" in text:
        score += 1
    if "-> list" in text or "-> List" in text:
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
    "coding_parser": score_coding_parser,
    "coding_concurrent": score_coding_concurrent,
    "coding_refactor": score_coding_refactor,
    "coding_lru_cache": score_coding_lru_cache,
    "coding_min_heap": score_coding_min_heap,
    "coding_async_debug": score_coding_async_debug,
    "coding_producer_consumer": score_coding_producer_consumer,
    "coding_api_design": score_coding_api_design,
    "coding_middleware": score_coding_middleware,
    "coding_regex": score_coding_regex,
    "coding_template_engine": score_coding_template_engine,
    "coding_binary_search": score_coding_binary_search,
    "coding_merge_intervals": score_coding_merge_intervals,
}

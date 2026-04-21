"""Test prompts for LLM quality benchmarking."""

from dataclasses import dataclass


@dataclass
class Prompt:
    name: str
    text: str
    description: str


PROMPTS = [
    Prompt(
        name="reasoning",
        text="I have a meeting at 3pm. It's currently 2:45pm. The meeting room is a 10-minute walk away, but it's raining heavily outside and I don't have an umbrella. Should I walk to the meeting room, or should I call in remotely? Explain your reasoning briefly.",
        description="Practical reasoning with constraints",
    ),
    Prompt(
        name="strawberry",
        text="How many r's are in the word 'strawberry'? Think step by step and count each one.",
        description="Character counting (common LLM failure)",
    ),
    Prompt(
        name="widgets",
        text="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Think carefully.",
        description="Cognitive reflection test",
    ),
    Prompt(
        name="coding",
        text="Write a Python function that takes a list of integers and returns the length of the longest consecutive elements sequence. For example, input [100, 4, 200, 1, 3, 2] should return 4 because the longest consecutive sequence is [1, 2, 3, 4]. The algorithm must run in O(n) time. Include the full function with type hints and a test case.",
        description="Algorithm implementation (O(n) consecutive sequence)",
    ),
    Prompt(
        name="haiku",
        text="Write a haiku about programming. Output ONLY the haiku — no title, no explanation, no additional text.",
        description="Constrained generation (5-7-5 syllables)",
    ),
    Prompt(
        name="bat_ball",
        text="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully.",
        description="Cognitive reflection (common wrong answer: $0.10)",
    ),
    Prompt(
        name="truth_liar",
        text="You meet two people, A and B. One always tells the truth, the other always lies. You don't know which is which. A says: 'B would tell you that I am the liar.' Who is the truth-teller and who is the liar? Explain your reasoning step by step.",
        description="Logic puzzle (truth-teller/liar)",
    ),
    Prompt(
        name="sql_injection",
        text="Write a Python Flask route handler for POST /login that safely authenticates a user against a SQLite database. Include protection against SQL injection, timing attacks, and brute force. Use parameterized queries.",
        description="Security-aware coding task",
    ),
    Prompt(
        name="coding_parser",
        text="Write a Python function `parse_csv_line(line: str, delimiter: str = ',') -> List[str]` that correctly handles: (1) quoted fields containing the delimiter, (2) escaped quotes (double-quote inside quoted field), (3) empty fields, (4) fields with leading/trailing whitespace inside quotes. Do NOT use the csv module — implement the parser from scratch. Include at least 5 test cases covering edge cases.",
        description="CSV parser from scratch (edge cases)",
    ),
    Prompt(
        name="coding_concurrent",
        text="Write a Python class `RateLimiter` that: (1) allows at most N requests per window of T seconds, (2) is thread-safe, (3) supports per-key rate limiting (different keys have independent counters), (4) automatically cleans up expired keys. Include type hints, docstrings, and tests using threading to verify thread safety.",
        description="Thread-safe rate limiter implementation",
    ),
    Prompt(
        name="coding_refactor",
        text="The following Python code has multiple bugs and design issues. Find and fix all of them. Return ONLY the corrected code with a comment above each fix explaining what was wrong.\n\n```python\ndef get_user_data(user_id, db):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    result = db.execute(query)\n    if result:\n        return result[0]\n    return None\n\ndef calculate_discount(price, customer_type, is_holiday):\n    if customer_type == 'premium':\n        discount = 0.2\n    if is_holiday:\n        discount = discount + 0.1\n    elif customer_type == 'regular':\n        discount = 0.1\n    else:\n        discount = 0\n    return price - price * discount\n\ndef process_orders(orders):\n    total = 0\n    for order in orders:\n        if order['status'] != 'cancelled':\n            total += order['amount']\n    avg = total / len(orders)\n    return {'total': total, 'average': avg}\n```",
        description="Bug finding and refactoring",
    ),
    Prompt(
        name="coding_lru_cache",
        text="Implement an LRU (Least Recently Used) cache in Python from scratch. Requirements: (1) Fixed capacity set at construction, (2) O(1) get and put operations, (3) When capacity is exceeded, evict the least recently used item, (4) get(key) returns the value or -1 if not found, and marks it as most recently used, (5) put(key, value) inserts or updates and marks as most recently used. Do NOT use OrderedDict or functools.lru_cache — use a doubly-linked list + hashmap. Include type hints and test cases.",
        description="LRU cache from scratch (data structures)",
    ),
    Prompt(
        name="coding_min_heap",
        text="Implement a MinHeap class in Python from scratch with these operations: (1) push(val) — insert a value, (2) pop() — remove and return the minimum, (3) peek() — return the minimum without removing, (4) heapify(list) — build a heap from an unsorted list in O(n) time, (5) size property. Do NOT use the heapq module. Include type hints and test cases covering edge cases: empty heap pop, duplicate values, single element, large numbers.",
        description="Min heap implementation (data structures)",
    ),
    Prompt(
        name="coding_async_debug",
        text="The following async Python code has multiple bugs. Find and fix ALL of them. Return ONLY the corrected code with a comment above each fix.\n\n```python\nimport asyncio\n\nasync def fetch_data(urls):\n    results = []\n    for url in urls:\n        response = await requests.get(url)\n        results.append(response.json())\n    return results\n\nasync def process_items(items):\n    async with asyncio.Lock() as lock:\n        for item in items:\n            await process_one(item)\n\nasync def retry_request(url, max_retries=3):\n    for i in range(max_retries):\n        try:\n            return await fetch(url)\n        except Exception:\n            continue\n\nasync def main():\n    task1 = asyncio.create_task(fetch_data(['http://a.com', 'http://b.com']))\n    task2 = asyncio.create_task(fetch_data(['http://c.com']))\n    result1 = task1.result()\n    result2 = task2.result()\n    return result1 + result2\n```",
        description="Async/concurrency bug fixing",
    ),
    Prompt(
        name="coding_producer_consumer",
        text="Implement a thread-safe bounded producer-consumer queue in Python. Requirements: (1) Fixed capacity buffer, (2) Multiple producer threads can call put(item) — blocks when full, (3) Multiple consumer threads can call get() — blocks when empty, (4) A shutdown() method that gracefully stops all waiting threads, (5) Proper handling of spurious wakeups. Use only threading primitives (Lock, Condition, etc.) — no queue.Queue. Include type hints and a test that runs 3 producers and 3 consumers concurrently.",
        description="Producer-consumer pattern (concurrency)",
    ),
    Prompt(
        name="coding_api_design",
        text="Design a Python REST API using Flask for a task management system. Implement these endpoints: (1) POST /tasks — create a task with title, description, priority (low/medium/high), (2) GET /tasks — list tasks with optional filtering by priority and status, and pagination (limit/offset), (3) GET /tasks/<id> — get a single task, (4) PUT /tasks/<id> — update a task, (5) DELETE /tasks/<id> — soft-delete a task. Requirements: proper input validation, consistent error response format with status codes, type annotations, and in-memory storage. Include example curl commands.",
        description="REST API design with error handling",
    ),
    Prompt(
        name="coding_middleware",
        text="Implement a Python middleware chain pattern for HTTP request processing. Build: (1) A Middleware base class with a process_request(request) -> response method, (2) A MiddlewareChain that chains multiple middlewares and passes the request through each in order, (3) Implement these middlewares: AuthMiddleware (checks Authorization header), RateLimitMiddleware (max N requests per minute per IP), LoggingMiddleware (logs method, path, status, duration), CorsMiddleware (adds CORS headers), (4) Each middleware can short-circuit by returning a response early, (5) Proper error handling — if a middleware raises, return 500. Include type hints and tests.",
        description="Middleware chain pattern (API design)",
    ),
    Prompt(
        name="coding_regex",
        text="Write a Python function `extract structured_data(text: str) -> dict` that extracts the following from unstructured text using regex: (1) All email addresses, (2) All phone numbers in formats: 555-123-4567, (555) 123-4567, 555.123.4567, +1 555 123 4567, (3) All URLs (http/https), (4) All dates in formats: YYYY-MM-DD, MM/DD/YYYY, Month DD, YYYY, (5) All dollar amounts: $1,234.56, $100, $.99. Return a dict with keys: emails, phones, urls, dates, amounts — each a list of strings. Include at least 5 test cases with mixed content.",
        description="Complex regex extraction from unstructured text",
    ),
    Prompt(
        name="coding_template_engine",
        text="Implement a simple template engine in Python. The function `render(template: str, context: dict) -> str` should support: (1) Variable substitution: {{ name }} — replaced by context['name'], (2) Conditional blocks: {% if condition %}...{% endif %} and {% if condition %}...{% else %}...{% endif %}, (3) For loops: {% for item in items %}...{{ item }}...{% endfor %}, (4) Nested access: {{ user.name }} for context with nested dicts, (5) Missing variables should render as empty string (not error). Do NOT use any template libraries. Include type hints and test cases covering: nested loops, if-else inside for loops, missing variables, special characters.",
        description="Template engine with loops and conditionals",
    ),
    Prompt(
        name="coding_binary_search",
        text="Implement these binary search variants in Python. Each function must handle edge cases correctly: (1) binary_search_exact(arr, target) -> int: return index of target in sorted array, or -1. Handle: empty array, single element, duplicates (return any matching index), target not present, (2) binary_search_first(arr, target) -> int: in sorted array with duplicates, return the FIRST index of target, or -1. Must be O(log n), (3) binary_search_closest(arr, target) -> int: return index of element closest to target. Handle: target outside array range. All functions must include type hints, be O(log n), and have at least 5 test cases each covering edge cases.",
        description="Binary search variants with edge cases",
    ),
    Prompt(
        name="coding_merge_intervals",
        text="Write a Python function `merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]` that takes a list of intervals and merges all overlapping ones. Requirements: (1) Handle unsorted input, (2) Handle fully contained intervals: (1,5) and (2,3) -> (1,5), (3) Handle touching intervals: (1,3) and (3,5) -> (1,5), (4) Handle empty input, (5) Handle single interval, (6) Return merged intervals sorted by start. Then also write `find_free_slots(busy: List[Tuple[int,int]], day_start: int, day_end: int) -> List[Tuple[int,int]]` that returns the free time slots in a day given busy intervals. Include type hints and comprehensive test cases.",
        description="Interval merging and free-slot finding",
    ),
]


def get_prompt(name: str) -> Prompt:
    """Get a prompt by name."""
    for p in PROMPTS:
        if p.name == name:
            return p
    raise ValueError(f"Unknown prompt: {name}. Available: {[p.name for p in PROMPTS]}")

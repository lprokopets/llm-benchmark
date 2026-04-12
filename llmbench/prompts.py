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
]


def get_prompt(name: str) -> Prompt:
    """Get a prompt by name."""
    for p in PROMPTS:
        if p.name == name:
            return p
    raise ValueError(f"Unknown prompt: {name}. Available: {[p.name for p in PROMPTS]}")

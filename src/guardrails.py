import re
from langchain_core.documents import Document

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions)",
    r"bypass\s+(your\s+)?(guardrails|rules|safety)",
    r"you\s+are\s+now\s+(dan|unrestricted|developer\s+mode)",
]
GREETING_PATTERN = re.compile(r"^(h+i+|h+e+y+|h+e+l+o+|good\s+(morning|afternoon|evening)|thanks+|thank\s+you)[!.,?\s]*$")

def check_input_guardrails(query: str) -> tuple[bool, str | None]:
    """
    Validate user query before retrieval.
    Returns (is_safe, refusal_message).
    """
    cleaned = query.strip()
    if len(cleaned) < 3:
        return False, "Query is too short. Please ask a specific financial question."

    lowered = cleaned.lower()

    if GREETING_PATTERN.fullmatch(lowered):
        return (
            False,
            "Hello! Ask a specific financial question about your indexed filings (e.g., revenue, EBIT margin, or segment performance).",
        )
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return (
                False,
                "Blocked by security guardrail: prompt-injection pattern detected.",
            )

    return True, None


def check_retrieval_guardrails(
    docs: list[Document],
    min_top_score: float = 0.05,
) -> tuple[bool, str | None]:
    """
    Ensure retrieved evidence is strong enough to answer without hallucinating.
    Returns (is_grounded, refusal_message).
    """
    if not docs:
        return (
            False,
            "No matching sections were found in the indexed financial filings for this query.",
        )

    top_score = float(docs[0].metadata.get("rerank_score", 0.0))
    if top_score < min_top_score:
        return (
            False,
            f"Insufficient evidence in the indexed filings to answer reliably (top relevance score: {top_score}).",
        )

    return True, None
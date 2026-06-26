"""retrieval.base - Retriever Protocol + Hit dataclass.

Every retriever (lexical / vector / graph) speaks this contract:

    from typing import Protocol
    class Retriever(Protocol):
        name: str
        def search(self, query: str, top_k: int = 5) -> list[Hit]: ...

Returns a uniform list[Hit], regardless of backend. qa_assistant
orchestrates multiple retrievers and converts hits to its own
Citation dataclass via hit_to_citation().
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Hit:
    """Uniform search hit returned by every Retriever."""
    id: str                 # unique doc id
    text: str               # snippet (200-300 chars recommended)
    score: float            # 0..1, higher is better
    source: str             # 'lexical' | 'vector' | 'graph' | 'manual'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Retriever(Protocol):
    """Protocol every retriever must satisfy."""
    name: str

    def search(self, query: str, top_k: int = 5, **kwargs) -> List[Hit]:
        ...


# ---------------------------------------------------------------------------
# Adapter: Hit -> qa_assistant.Citation (avoid cyclic import)
# ---------------------------------------------------------------------------


class CitationLike:
    """Lightweight stand-in matching qa_assistant.Citation fields.

    The qa_assistant module defines its own Citation dataclass, but
    importing it from retrieval/ would create a cycle. hit_to_citation
    is the explicit bridge: callers pass a Citation class in, and we
    produce an instance with the right fields.
    """
    pass


def hit_to_citation(hit: Hit, citation_cls=None):
    """Convert a Hit into a qa_assistant.Citation (or duck-typed equivalent).

    If citation_cls is None, returns a dict (qa_assistant normally
    reconstructs from dict in some code paths; tests can pass their
    own minimal stand-in).
    """
    payload = {
        "source": hit.source + ":" + hit.id,
        "date": hit.metadata.get("date", ""),
        "score": hit.score,
        "snippet": hit.text[:200].replace("\n", " "),
    }
    if citation_cls is None:
        return payload
    try:
        return citation_cls(**payload)
    except TypeError:
        # Citation expects different field names; fall back to kwargs
        return citation_cls(
            source=payload["source"],
            date=payload["date"],
            score=payload["score"],
            snippet=payload["snippet"],
        )

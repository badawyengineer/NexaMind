from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import BaseModel, ConfigDict, Field


class WebSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=8)


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    if not query.strip():
        return []
    try:
        return [
            {
                "title": str(r.get("title", "")),
                "snippet": str(r.get("body", "")),
                "url": str(r.get("href", "")),
            }
            for r in DDGS().text(query, max_results=max_results)
        ]
    except DDGSException as exc:
        return [{"error": f"Search unavailable: {type(exc).__name__}"}]

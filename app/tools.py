from typing import Any, Dict, List, Literal, Optional

from agents import RunContextWrapper, function_tool
from tinydb import Query
from veris_ai import veris

from .db import db
from .schema import CRMRunContext


def _ci_contains(value: str, keyword: str) -> bool:
    if not isinstance(value, str):
        return False
    return keyword.lower() in value.lower()


def _get_lead_by_id(lead_id: int) -> Optional[Dict[str, Any]]:
    lead_query = Query()
    return db.get(lead_query.id == lead_id)


@function_tool
@veris.mock()
def lookup_lead(ctx: RunContextWrapper[CRMRunContext], query: str) -> Optional[Dict[str, Any]]:
    """Find best single lead whose name or contact contains the query (case-insensitive).

    Args:
        query: Free-text search over name and contact fields.
    """

    lead_query = Query()
    matches = db.search(
        (lead_query.name.test(lambda v: _ci_contains(v, query)))
        | (lead_query.contact.test(lambda v: _ci_contains(v, query)))
    )

    # Prefer exact name match if any; otherwise first match
    best: Optional[Dict[str, Any]] = None
    for m in matches:
        if isinstance(m.get("name"), str) and m["name"].strip().lower() == query.strip().lower():
            best = m
            break
    if best is None:
        best = matches[0] if matches else None

    ctx.context.tool_calls.append({
        "name": "lookup_lead",
        "args": {"query": query},
        "result": best,
    })
    return best


@function_tool
@veris.mock()
def write_lead_update(
    ctx: RunContextWrapper[CRMRunContext],
    lead_id: int,
    note: str,
    status: Literal["new", "working", "qualified", "disqualified"],
) -> Dict[str, Any]:
    """Append a note and set the status for the given lead, returning the updated record.

    Args:
        lead_id: Lead identifier.
        note: Note to append to the lead's notes list.
        status: New status value.
    """

    lead_query = Query()
    existing = _get_lead_by_id(lead_id)
    if existing is None:
        msg = f"Lead with id {lead_id} not found"
        ctx.context.tool_calls.append({
            "name": "write_lead_update",
            "args": {"lead_id": lead_id, "note": note, "status": status},
            "error": msg,
        })
        raise ValueError(msg)

    new_notes = list(existing.get("notes", []))
    new_notes.append(note)

    db.update({"status": status, "notes": new_notes}, lead_query.id == lead_id)
    updated = _get_lead_by_id(lead_id)

    ctx.context.tool_calls.append({
        "name": "write_lead_update",
        "args": {"lead_id": lead_id, "note": note, "status": status},
        "result": updated,
    })

    # Surface for observability bundle
    ctx.context.data["updated_lead"] = updated
    return updated  # type: ignore[return-value]


@function_tool
@veris.mock()
def get_leads(ctx: RunContextWrapper[CRMRunContext], keyword: str) -> List[Dict[str, Any]]:
    """Return all leads where name|contact|industry|status contains keyword (case-insensitive)."""

    lead_query = Query()
    results = db.search(
        (lead_query.name.test(lambda v: _ci_contains(v, keyword)))
        | (lead_query.contact.test(lambda v: _ci_contains(v, keyword)))
        | (lead_query.industry.test(lambda v: _ci_contains(v, keyword)))
        | (lead_query.status.test(lambda v: _ci_contains(v, keyword)))
    )

    ctx.context.tool_calls.append({
        "name": "get_leads",
        "args": {"keyword": keyword},
        "result_count": len(results),
    })

    # Expose in response for confirmations/browsing
    ctx.context.data["matches"] = results
    return results


@function_tool
@veris.mock()
def score_lead_industry(ctx: RunContextWrapper[CRMRunContext], industry: str) -> int:
    """Return a simple heuristic score for an industry.

    finance=8, retail=5, software=6, other=4; default=3
    """

    scores = {"finance": 8, "retail": 5, "software": 6, "other": 4}
    score = scores.get((industry or "").lower(), 3)

    ctx.context.tool_calls.append({
        "name": "score_lead_industry",
        "args": {"industry": industry},
        "result": score,
    })
    return score



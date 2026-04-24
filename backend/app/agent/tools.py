"""
LangGraph agent tools for HCP interaction management.

Five tools that the ReAct agent can invoke:
1. log_interaction     – persist a new interaction to PostgreSQL
2. edit_interaction    – modify an existing interaction
3. search_hcp          – search the HCP database
4. get_interaction_history – fetch past interactions for an HCP
5. suggest_follow_up   – AI-generated follow-up recommendations
"""

import json
import datetime
from langchain_core.tools import tool
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.hcp import HCP
from app.models.interaction import Interaction


def _get_db() -> Session:
    """Create a fresh DB session for tool use (outside FastAPI request lifecycle)."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Tool 1: log_interaction
# ---------------------------------------------------------------------------
@tool
def log_interaction(
    hcp_id,
    interaction_type: str,
    interaction_date: str,
    topics_discussed: str,
    hcp_sentiment: str = "Neutral",
    outcomes: str = "",
    follow_up_actions: str = "",
    attendees: str = "[]",
    materials_shared: str = "[]",
    samples_distributed: str = "[]",
    rep_name: str = "",
    ai_summary: str = "",
) -> str:
    """Log a new HCP interaction to the CRM database.

    Use this tool when the field representative has provided enough details
    about their meeting/call with an HCP and wants to save it. The tool will
    persist the interaction record and return a confirmation.

    Args:
        hcp_id: The ID of the HCP from the database (accepts int or string).
        interaction_type: Type of interaction - Meeting, Call, Video, or Email.
        interaction_date: Date of the interaction in YYYY-MM-DD format.
        topics_discussed: Key discussion points from the interaction.
        hcp_sentiment: Observed sentiment - Positive, Neutral, or Negative.
        outcomes: Key outcomes or agreements reached.
        follow_up_actions: Next steps or tasks to follow up on.
        attendees: JSON string list of attendee names, e.g. '["Dr. Smith", "Nurse Lee"]'.
        materials_shared: JSON string list of materials, e.g. '[{"name": "Brochure X"}]'.
        samples_distributed: JSON string list of samples, e.g. '[{"name": "Drug A", "quantity": 2}]'.
        rep_name: Name of the field representative logging the interaction.
        ai_summary: A concise AI-generated summary of the interaction.
    """
    # Convert hcp_id to int if it's a string
    if isinstance(hcp_id, str):
        hcp_id = int(hcp_id)
    elif not isinstance(hcp_id, int):
        hcp_id = int(hcp_id)

    db = _get_db()
    try:
        record = Interaction(
            hcp_id=hcp_id,
            interaction_type=interaction_type,
            interaction_date=datetime.date.fromisoformat(interaction_date),
            topics_discussed=topics_discussed,
            hcp_sentiment=hcp_sentiment,
            outcomes=outcomes,
            follow_up_actions=follow_up_actions,
            attendees=json.loads(attendees)
            if isinstance(attendees, str)
            else attendees,
            materials_shared=json.loads(materials_shared)
            if isinstance(materials_shared, str)
            else materials_shared,
            samples_distributed=json.loads(samples_distributed)
            if isinstance(samples_distributed, str)
            else samples_distributed,
            rep_name=rep_name,
            ai_summary=ai_summary,
            ai_extracted_entities={
                "topics": topics_discussed,
                "sentiment": hcp_sentiment,
                "materials": materials_shared,
                "samples": samples_distributed,
            },
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return json.dumps(
            {
                "status": "success",
                "interaction_id": record.id,
                "message": f"Interaction #{record.id} logged successfully for HCP #{hcp_id} on {interaction_date}.",
            }
        )
    except Exception as e:
        db.rollback()
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 2: edit_interaction
# ---------------------------------------------------------------------------
@tool
def edit_interaction(
    interaction_id: int,
    topics_discussed: str = None,
    hcp_sentiment: str = None,
    outcomes: str = None,
    follow_up_actions: str = None,
    interaction_type: str = None,
    materials_shared: str = None,
    samples_distributed: str = None,
    ai_summary: str = None,
) -> str:
    """Edit an existing HCP interaction record in the CRM database.

    Use this tool when the field representative wants to modify a previously
    logged interaction. Only the provided fields will be updated; others
    remain unchanged.

    Args:
        interaction_id: The ID of the interaction to edit.
        topics_discussed: Updated discussion points (optional).
        hcp_sentiment: Updated sentiment — Positive, Neutral, or Negative (optional).
        outcomes: Updated outcomes (optional).
        follow_up_actions: Updated follow-up actions (optional).
        interaction_type: Updated interaction type (optional).
        materials_shared: Updated materials as JSON string (optional).
        samples_distributed: Updated samples as JSON string (optional).
        ai_summary: Updated AI summary (optional).
    """
    db = _get_db()
    try:
        record = db.query(Interaction).filter(Interaction.id == interaction_id).first()
        if not record:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Interaction #{interaction_id} not found.",
                }
            )

        updates = {}
        if topics_discussed is not None:
            record.topics_discussed = topics_discussed
            updates["topics_discussed"] = topics_discussed
        if hcp_sentiment is not None:
            record.hcp_sentiment = hcp_sentiment
            updates["hcp_sentiment"] = hcp_sentiment
        if outcomes is not None:
            record.outcomes = outcomes
            updates["outcomes"] = outcomes
        if follow_up_actions is not None:
            record.follow_up_actions = follow_up_actions
            updates["follow_up_actions"] = follow_up_actions
        if interaction_type is not None:
            record.interaction_type = interaction_type
            updates["interaction_type"] = interaction_type
        if materials_shared is not None:
            record.materials_shared = (
                json.loads(materials_shared)
                if isinstance(materials_shared, str)
                else materials_shared
            )
            updates["materials_shared"] = materials_shared
        if samples_distributed is not None:
            record.samples_distributed = (
                json.loads(samples_distributed)
                if isinstance(samples_distributed, str)
                else samples_distributed
            )
            updates["samples_distributed"] = samples_distributed
        if ai_summary is not None:
            record.ai_summary = ai_summary
            updates["ai_summary"] = ai_summary

        db.commit()
        return json.dumps(
            {
                "status": "success",
                "interaction_id": interaction_id,
                "message": f"Interaction #{interaction_id} updated successfully.",
                "updated_fields": list(updates.keys()),
            }
        )
    except Exception as e:
        db.rollback()
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 3: search_hcp
# ---------------------------------------------------------------------------
@tool
def search_hcp(query: str) -> str:
    """Search the HCP database by name, specialty, institution, or NPI number.

    Use this tool when you need to find an HCP in the database, for example
    when the rep mentions a doctor's name and you need their database ID.

    Args:
        query: Search term - can be a name, specialty, institution, or NPI.
    """
    db = _get_db()
    try:
        # Handle multi-word queries by splitting into parts
        # e.g., "Rajesh Sharma" should match first_name="Rajesh", last_name="Sharma"
        query_parts = query.strip().split()

        # Build filter conditions
        from sqlalchemy import or_

        conditions = []
        for part in query_parts:
            part_pattern = f"%{part}%"
            conditions.append(HCP.first_name.ilike(part_pattern))
            conditions.append(HCP.last_name.ilike(part_pattern))
            conditions.append(HCP.specialty.ilike(part_pattern))
            conditions.append(HCP.institution.ilike(part_pattern))
            conditions.append(HCP.npi_number.ilike(part_pattern))

        # Apply OR filter for all conditions
        results = db.query(HCP).filter(or_(*conditions)).limit(10).all()
        if not results:
            return json.dumps(
                {
                    "status": "no_results",
                    "message": f"No HCPs found matching '{query}'.",
                    "results": [],
                }
            )
        hcp_list = [
            {
                "id": h.id,
                "name": f"Dr. {h.first_name} {h.last_name}",
                "specialty": h.specialty,
                "institution": h.institution,
                "npi_number": h.npi_number,
                "tier": h.tier,
            }
            for h in results
        ]
        return json.dumps(
            {
                "status": "success",
                "count": len(hcp_list),
                "results": hcp_list,
            }
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 4: get_interaction_history
# ---------------------------------------------------------------------------
@tool
def get_interaction_history(hcp_id, limit: int = 5) -> str:
    """Retrieve past interaction history for a specific HCP.

    Use this tool to provide context about previous meetings/calls with
    an HCP before logging a new interaction. This helps the agent and rep
    understand the relationship history.

    Args:
        hcp_id: The database ID of the HCP (accepts int or string).
        limit: Maximum number of past interactions to return (default 5).
    """
    # Convert to int if string
    if isinstance(hcp_id, str):
        hcp_id = int(hcp_id)
    elif not isinstance(hcp_id, int):
        hcp_id = int(hcp_id)

    db = _get_db()
    try:
        records = (
            db.query(Interaction)
            .filter(Interaction.hcp_id == hcp_id)
            .order_by(Interaction.interaction_date.desc())
            .limit(limit)
            .all()
        )
        if not records:
            return json.dumps(
                {
                    "status": "no_history",
                    "message": f"No past interactions found for HCP #{hcp_id}.",
                    "interactions": [],
                }
            )
        history = [
            {
                "id": r.id,
                "date": str(r.interaction_date),
                "type": r.interaction_type,
                "sentiment": r.hcp_sentiment,
                "topics": r.topics_discussed or "",
                "outcomes": r.outcomes or "",
                "ai_summary": r.ai_summary or "",
            }
            for r in records
        ]
        return json.dumps(
            {
                "status": "success",
                "hcp_id": hcp_id,
                "count": len(history),
                "interactions": history,
            }
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Tool 5: suggest_follow_up
# ---------------------------------------------------------------------------
@tool
def suggest_follow_up(
    hcp_id,
    topics_discussed: str,
    hcp_sentiment: str,
    outcomes: str = "",
) -> str:
    """Generate AI-powered follow-up suggestions based on the interaction.

    Use this tool after logging an interaction to provide the field rep with
    intelligent next-step recommendations. Considers the HCP profile, recent
    history, and the current interaction details.

    Args:
        hcp_id: The database ID of the HCP (accepts int or string).
        topics_discussed: What was discussed in the interaction.
        hcp_sentiment: The observed sentiment - Positive, Neutral, or Negative.
        outcomes: Key outcomes or agreements from the interaction.
    """
    # Convert to int if string
    if isinstance(hcp_id, str):
        hcp_id = int(hcp_id)
    elif not isinstance(hcp_id, int):
        hcp_id = int(hcp_id)

    db = _get_db()
    try:
        hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
        hcp_name = f"Dr. {hcp.first_name} {hcp.last_name}" if hcp else f"HCP #{hcp_id}"
        hcp_specialty = hcp.specialty if hcp else "Unknown"
        hcp_tier = hcp.tier if hcp else "C"

        # Generate contextual suggestions based on interaction data
        suggestions = []

        # Sentiment-based suggestions
        if hcp_sentiment == "Positive":
            suggestions.append(
                f"Schedule follow-up meeting with {hcp_name} within 2 weeks to maintain momentum"
            )
            suggestions.append(
                f"Send relevant Phase III clinical data via {hcp.preferred_channel or 'email'}"
            )
        elif hcp_sentiment == "Negative":
            suggestions.append(
                f"Address {hcp_name}'s concerns with targeted clinical evidence within 1 week"
            )
            suggestions.append(
                "Consult with Medical Affairs team for additional support materials"
            )
        else:
            suggestions.append(
                f"Follow up with {hcp_name} in 3 weeks with updated product information"
            )

        # Tier-based suggestions
        if hcp_tier == "A":
            suggestions.append(
                f"Consider inviting {hcp_name} to upcoming advisory board meeting"
            )
            suggestions.append(
                f"Share exclusive KOL engagement opportunities with {hcp_name}"
            )
        elif hcp_tier == "B":
            suggestions.append(
                f"Invite {hcp_name} to next regional medical education event"
            )

        # Topic-based suggestions
        if topics_discussed:
            suggestions.append(
                f"Prepare updated materials on: {topics_discussed[:100]}"
            )

        return json.dumps(
            {
                "status": "success",
                "hcp_name": hcp_name,
                "specialty": hcp_specialty,
                "suggestions": suggestions[:5],  # cap at 5
            }
        )
    finally:
        db.close()


# Export all tools as a list for LangGraph
ALL_TOOLS = [
    log_interaction,
    edit_interaction,
    search_hcp,
    get_interaction_history,
    suggest_follow_up,
]

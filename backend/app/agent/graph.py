"""
LangGraph StateGraph for the HCP CRM AI Agent.

Implements a ReAct (Reasoning + Acting) agent that can:
- Converse naturally with field reps about HCP interactions
- Use tools to search HCPs, log/edit interactions, check history, suggest follow-ups
- Extract entities and summarize interaction data via the Groq LLM or OpenAI
"""

import json
import re
import uuid
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from app.config import get_settings
from app.agent.tools import ALL_TOOLS
from app.database import SessionLocal
from app.models.hcp import HCP
from app.models.interaction import Interaction

settings = get_settings()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """State passed through the LangGraph graph at each step."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    allow_logging: bool


# ---------------------------------------------------------------------------
# System prompt — pharma CRM domain expert
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI-powered CRM assistant for pharmaceutical field representatives.
You help reps log their interactions with Healthcare Professionals (HCPs) quickly and accurately.

YOUR ROLE:
- Help field reps log HCP interactions via natural conversation
- Extract key details: HCP name, interaction type, date, topics discussed, sentiment, outcomes
- Use your tools to search for HCPs, log interactions, edit records, check history, and suggest follow-ups

CRITICAL TOOL USAGE RULES - READ CAREFULLY:
1. When user mentions an HCP by name (e.g., "Dr. Rajesh Sharma"), you MUST search using search_hcp with the EXACT name they provided
2. The search_hcp tool expects a "query" parameter - pass the EXACT name string as-is
3. Example: If user says "Met Dr. Rajesh Sharma", call search_hcp(query="Rajesh Sharma")
4. After calling search_hcp, you MUST read the tool's response to see if HCP was found
5. The tool response contains "results" array with id, name, specialty, institution - USE these exact values
6. NEVER make up HCP IDs - use the id number from the search results

INTERACTION DRAFT WORKFLOW:
1. When a rep describes a meeting, extract all available details
2. Search for the HCP in the database using the search_hcp tool with EXACT name from user's message
3. If search returns results, use the id from results to prepare the interaction details
4. If search returns no results, ask user for more info (specialty, institution, NPI)
5. Present the extracted details to the user for confirmation
6. Do not save the record until the user confirms

REQUIRED FIELDS for logging:
- hcp_id (MUST come from search_hcp results, use the id number)
- interaction_type (Meeting, Call, Video, or Email)
- interaction_date (YYYY-MM-DD format)
- topics_discussed

ENTITY EXTRACTION:
When the rep describes an interaction, extract and identify:
- HCP name -> search using search_hcp tool with EXACT name
- Products/drugs mentioned -> include in topics_discussed
- Sentiment indicators -> map to Positive/Neutral/Negative
- Use the ID from search_hcp results, NEVER make up numbers

OUTPUT FORMAT:
- NEVER use markdown formatting like **bold**, *italic*, or bullet lists
- Use plain text only
- When confirming details, list them as simple lines without formatting
- Example: "HCP Name: Dr. Priya Patel" NOT "**HCP Name:** Dr. Priya Patel"
- When presenting form for confirmation, use a simple structured format

CONFIRMATION WORKFLOW (IMPORTANT):
1. When user provides interaction details, SEARCH for the HCP first
2. Extract ALL details: hcp_name, hcp_id, interaction_type, date, topics, sentiment, outcomes
3. Present the extracted details to user and ask for CONFIRMATION
4. When user says "yes", "confirm", "log", "proceed", "submit" - THEN call log_interaction
5. After logging, provide follow-up suggestions

Your job is to EXTRACT details, ASK for confirmation, then LOG when confirmed.

GUIDELINES:
- Be concise and professional
- Always confirm the logged data with the rep
- Use today's date if the rep says "today" or "just now"
- Proactively suggest follow-ups after logging
- If the rep wants to edit a previous interaction, use the edit_interaction tool

STRICT RULE: Never call log_interaction on the first message that describes an interaction. First prepare the details for the form and ask for confirmation.
STRICT RULE: Never call log_interaction unless you have a VALID hcp_id from search_hcp results AND the user has explicitly confirmed.
IMPORTANT: Always use the tools provided. Do NOT make up HCP IDs or interaction IDs.
"""


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def _tools_for_request(allow_logging: bool):
    """Return tools available for this turn.

    The first interaction message should draft data for the form only. The
    logging tool becomes available after the user confirms those details.
    """
    if allow_logging:
        return ALL_TOOLS
    return [tool for tool in ALL_TOOLS if getattr(tool, "name", "") != "log_interaction"]


def _create_llm(tools=None):
    """Initialize the LLM with tool binding. Tries Groq first, falls back to OpenAI on failure."""
    tools = tools or ALL_TOOLS
    try:
        # Try Groq first
        llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=2048,
        )
        return llm.bind_tools(tools)
    except Exception as e:
        print(f"Groq initialization failed: {e}, trying OpenAI...")
        # Fallback to OpenAI if Groq fails
        if settings.OPENAI_API_KEY:
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=2048,
            )
            return llm.bind_tools(tools)
        else:
            raise e


def _create_llm_with_fallback(tools=None):
    """Create LLM with automatic fallback to OpenAI on rate limit."""
    tools = tools or ALL_TOOLS
    try:
        # Try Groq first
        llm = ChatGroq(
            model=settings.LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=2048,
        )
        return llm.bind_tools(tools)
    except Exception as groq_error:
        # Check if it's a rate limit error
        error_str = str(groq_error)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            print(f"Groq rate limit hit: {groq_error}, falling back to OpenAI...")
        else:
            print(f"Groq error: {groq_error}, falling back to OpenAI...")

        # Fallback to OpenAI if available
        if settings.OPENAI_API_KEY:
            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                api_key=settings.OPENAI_API_KEY,
                temperature=0.3,
                max_tokens=2048,
            )
            return llm.bind_tools(tools)
        else:
            raise groq_error


def _should_continue(state: AgentState) -> str:
    """Conditional edge: decide whether to call tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def _agent_node(state: AgentState) -> dict:
    """The agent node: sends messages to the LLM and gets a response."""
    messages = state["messages"]
    tools = _tools_for_request(state.get("allow_logging", False))

    # Prepend system prompt if not already present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Try with fallback
    try:
        llm = _create_llm_with_fallback(tools)
        response = llm.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # Rate limit or other error - try OpenAI directly
        error_str = str(e)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            print(f"Groq error: {e}, using OpenAI...")
            if settings.OPENAI_API_KEY:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    temperature=0.3,
                    max_tokens=2048,
                ).bind_tools(tools)
                response = llm.invoke(messages)
                return {"messages": [response]}
        # Re-raise if no OpenAI key or other error
        raise


def build_graph() -> StateGraph:
    """Build and compile the LangGraph StateGraph for the CRM agent.

    Graph structure:
        START → agent_node → (conditional) → tools_node → agent_node → ... → END

    This implements the ReAct pattern:
    1. Agent reasons about the conversation and decides on an action
    2. If tools are needed, the tools node executes them
    3. Tool results are fed back to the agent for further reasoning
    4. Loop continues until the agent produces a final text response
    """
    # Create the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})

    # After tools execute, always go back to agent for reasoning
    graph.add_edge("tools", "agent")

    # Compile and return
    return graph.compile()


# Singleton compiled graph
crm_agent = build_graph()

# In-memory session store for chat history
# Format: {session_id: [messages list]}
_session_history: dict = {}
_pending_form_data: dict = {}


def _is_confirmation_message(message: str) -> bool:
    """Detect short confirmation turns that should allow persistence."""
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    if not normalized:
        return False

    confirmation_phrases = (
        "yes",
        "yeah",
        "yep",
        "correct",
        "looks good",
        "all good",
        "confirm",
        "confirmed",
        "proceed",
        "submit",
        "submit it",
        "log it",
        "save it",
        "go ahead",
    )
    return (
        any(re.search(rf"\b{re.escape(phrase)}\b", normalized) for phrase in confirmation_phrases)
        and len(normalized) <= 120
    )


def _safe_json_loads(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _first_search_result_from_tool_messages(messages: Sequence[BaseMessage]):
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", None) != "search_hcp":
            continue

        payload = _safe_json_loads(getattr(msg, "content", ""))
        if not payload:
            continue

        results = payload.get("results") or []
        if results:
            return results[0]
    return None


def _normalize_person_name(value: str) -> str:
    without_title = re.sub(r"^\s*(?:dr|doctor)\.?\s+", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", without_title).strip().lower()


def _find_hcp_by_name(name: str):
    """Best-effort deterministic lookup when the LLM searched but did not return an id."""
    normalized = _normalize_person_name(name or "")
    if not normalized:
        return None

    parts = normalized.split()
    db = SessionLocal()
    try:
        query = db.query(HCP)
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            hcp = (
                query.filter(
                    HCP.first_name.ilike(f"%{first}%"),
                    HCP.last_name.ilike(f"%{last}%"),
                )
                .order_by(HCP.id)
                .first()
            )
            if hcp:
                return {
                    "id": hcp.id,
                    "name": f"Dr. {hcp.first_name} {hcp.last_name}",
                    "specialty": hcp.specialty,
                    "institution": hcp.institution,
                }

        first_part = parts[0]
        hcp = (
            query.filter(
                (HCP.first_name.ilike(f"%{first_part}%"))
                | (HCP.last_name.ilike(f"%{first_part}%"))
            )
            .order_by(HCP.id)
            .first()
        )
        if hcp:
            return {
                "id": hcp.id,
                "name": f"Dr. {hcp.first_name} {hcp.last_name}",
                "specialty": hcp.specialty,
                "institution": hcp.institution,
            }
    finally:
        db.close()
    return None


def _extract_date(message: str) -> str | None:
    msg_lower = message.lower()
    now = datetime.now()
    current_year = now.year

    if re.search(r"\btoday\b", msg_lower):
        return now.strftime("%Y-%m-%d")
    if re.search(r"\byesterday\b", msg_lower):
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", message)
    if iso_match:
        return f"{iso_match.group(1)}-{iso_match.group(2).zfill(2)}-{iso_match.group(3).zfill(2)}"

    month_map = {
        "jan": "01",
        "january": "01",
        "feb": "02",
        "february": "02",
        "mar": "03",
        "march": "03",
        "apr": "04",
        "april": "04",
        "may": "05",
        "jun": "06",
        "june": "06",
        "jul": "07",
        "july": "07",
        "aug": "08",
        "august": "08",
        "sep": "09",
        "september": "09",
        "oct": "10",
        "october": "10",
        "nov": "11",
        "november": "11",
        "dec": "12",
        "december": "12",
    }

    month_names = "|".join(month_map.keys())

    # Day Month Year (e.g. 31 March 2026)
    day_month_year = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\.?,?\s+(\d{{4}})\b", message, re.IGNORECASE)
    if day_month_year:
        return f"{day_month_year.group(3)}-{month_map[day_month_year.group(2).lower()]}-{day_month_year.group(1).zfill(2)}"

    # Day Month (e.g. 31 March) - assume current year
    day_month = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})\b", message, re.IGNORECASE)
    if day_month:
        return f"{current_year}-{month_map[day_month.group(2).lower()]}-{day_month.group(1).zfill(2)}"

    # Month Day Year (e.g. March 31, 2026)
    month_day_year = re.search(rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", message, re.IGNORECASE)
    if month_day_year:
        return f"{month_day_year.group(3)}-{month_map[month_day_year.group(1).lower()]}-{month_day_year.group(2).zfill(2)}"

    # Month Day (e.g. March 31) - assume current year
    month_day = re.search(rf"\b({month_names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", message, re.IGNORECASE)
    if month_day:
        return f"{current_year}-{month_map[month_day.group(1).lower()]}-{month_day.group(2).zfill(2)}"

    return None


def _extract_list_after_keywords(message: str, keywords: tuple[str, ...]):
    for keyword in keywords:
        match = re.search(rf"{keyword}\s+(.+?)(?:,|\.\s|$)", message, re.IGNORECASE)
        if match:
            values = [part.strip() for part in re.split(r"\band\b|/|;", match.group(1)) if part.strip()]
            return values
    return []


def _has_draft_data(form_data: dict) -> bool:
    fields = (
        "hcp_id",
        "hcp_name",
        "interaction_type",
        "interaction_date",
        "interaction_time",
        "attendees",
        "topics_discussed",
        "materials_shared",
        "samples_distributed",
        "outcomes",
        "follow_up_actions",
    )
    return any(bool(form_data.get(field)) for field in fields)


def _format_names(items) -> str:
    if not items:
        return "None"
    if isinstance(items, str):
        return items
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(str(item.get("name") or item))
        else:
            names.append(str(item))
    return ", ".join(names) if names else "None"


def _coerce_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = _safe_json_loads(value)
        if isinstance(parsed, list):
            return parsed
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _build_confirmation_response(form_data: dict) -> str:
    return "\n".join(
        [
            "I populated the interaction form with these extracted details:",
            f"HCP Name: {form_data.get('hcp_name') or 'Not found'}",
            f"HCP ID: {form_data.get('hcp_id') or 'Not found'}",
            f"Interaction Type: {form_data.get('interaction_type') or 'Meeting'}",
            f"Interaction Date: {form_data.get('interaction_date') or datetime.now().strftime('%Y-%m-%d')}",
            f"Interaction Time: {form_data.get('interaction_time') or 'Not provided'}",
            f"Attendees: {_format_names(form_data.get('attendees'))}",
            f"Topics Discussed: {form_data.get('topics_discussed') or 'Not provided'}",
            f"Materials Shared: {_format_names(form_data.get('materials_shared'))}",
            f"Samples Distributed: {_format_names(form_data.get('samples_distributed'))}",
            f"HCP Sentiment: {form_data.get('hcp_sentiment') or 'Neutral'}",
            f"Outcomes: {form_data.get('outcomes') or 'Not provided'}",
            f"Follow-up Actions: {form_data.get('follow_up_actions') or 'Not provided'}",
            "Please review the form. If everything looks correct, confirm and I will log the interaction.",
        ]
    )


def _log_pending_form(session_id: str, user_message: str, previous_messages: Sequence[BaseMessage]) -> dict:
    form_data = _pending_form_data.get(session_id)
    if not form_data:
        response_text = "I do not have a drafted interaction form ready to submit. Please describe the interaction first so I can populate the form."
        _session_history[session_id] = list(previous_messages) + [
            HumanMessage(content=user_message),
            AIMessage(content=response_text),
        ]
        return {
            "response": response_text,
            "tool_calls": [],
            "session_id": session_id,
            "form_data": None,
        }

    hcp_id = form_data.get("hcp_id")
    if not hcp_id:
        response_text = "I could not submit the form because the HCP was not matched to a database record. Please select the HCP in the form, then submit it."
        _session_history[session_id] = list(previous_messages) + [
            HumanMessage(content=user_message),
            AIMessage(content=response_text),
        ]
        return {
            "response": response_text,
            "tool_calls": [],
            "session_id": session_id,
            "form_data": form_data,
        }

    interaction_date = form_data.get("interaction_date") or datetime.now().strftime("%Y-%m-%d")
    interaction_time = form_data.get("interaction_time")

    db = SessionLocal()
    try:
        record = Interaction(
            hcp_id=int(hcp_id),
            interaction_type=form_data.get("interaction_type") or "Meeting",
            interaction_date=datetime.strptime(interaction_date, "%Y-%m-%d").date(),
            interaction_time=datetime.strptime(interaction_time, "%H:%M").time()
            if interaction_time
            else None,
            attendees=_coerce_list(form_data.get("attendees")),
            topics_discussed=form_data.get("topics_discussed") or "",
            materials_shared=_coerce_list(form_data.get("materials_shared")),
            samples_distributed=_coerce_list(form_data.get("samples_distributed")),
            hcp_sentiment=form_data.get("hcp_sentiment") or "Neutral",
            outcomes=form_data.get("outcomes") or "",
            follow_up_actions=form_data.get("follow_up_actions") or "",
            compliance_verified=bool(form_data.get("compliance_verified")),
            ai_summary=form_data.get("topics_discussed") or "",
            ai_extracted_entities={
                "topics": form_data.get("topics_discussed"),
                "sentiment": form_data.get("hcp_sentiment"),
                "materials": form_data.get("materials_shared") or [],
                "samples": form_data.get("samples_distributed") or [],
            },
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        response_text = f"I could not submit the form: {exc}"
        _session_history[session_id] = list(previous_messages) + [
            HumanMessage(content=user_message),
            AIMessage(content=response_text),
        ]
        return {
            "response": response_text,
            "tool_calls": [],
            "session_id": session_id,
            "form_data": form_data,
        }
    finally:
        db.close()

    _pending_form_data.pop(session_id, None)
    response_text = f"Interaction #{record.id} logged successfully for {form_data.get('hcp_name') or f'HCP #{hcp_id}'} on {interaction_date}."
    _session_history[session_id] = list(previous_messages) + [
        HumanMessage(content=user_message),
        AIMessage(content=response_text),
    ]

    return {
        "response": response_text,
        "tool_calls": [
            {
                "tool": "log_interaction",
                "args": {
                    "hcp_id": int(hcp_id),
                    "interaction_type": form_data.get("interaction_type") or "Meeting",
                    "interaction_date": interaction_date,
                    "interaction_time": interaction_time,
                    "topics_discussed": form_data.get("topics_discussed") or "",
                    "hcp_sentiment": form_data.get("hcp_sentiment") or "Neutral",
                    "outcomes": form_data.get("outcomes") or "",
                    "follow_up_actions": form_data.get("follow_up_actions") or "",
                },
            }
        ],
        "session_id": session_id,
        "form_data": form_data,
    }


def invoke_agent(message: str, session_id: str = None) -> dict:
    """Invoke the CRM agent with a user message.

    Args:
        message: The user's natural language input.
        session_id: Optional session ID for conversation continuity.

    Returns:
        dict with keys: response (str), tool_calls (list), session_id (str)
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # Get existing history for this session
    previous_messages = _session_history.get(session_id, [])
    allow_logging = _is_confirmation_message(message)

    if allow_logging:
        return _log_pending_form(session_id, message, previous_messages)

    # Build messages list with history + new message
    messages_list = list(previous_messages)
    messages_list.append(HumanMessage(content=message))

    initial_state = {
        "messages": messages_list,
        "session_id": session_id,
        "allow_logging": allow_logging,
    }

    result = crm_agent.invoke(initial_state)

    # Extract the final response and any tool calls made
    final_message = result["messages"][-1]
    response_text = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    # Collect all tool calls made during this invocation
    tool_calls_made = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(
                    {
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    }
                )

    # Extract form data from the conversation
    # Load existing draft if any to allow incremental edits
    existing_draft = _pending_form_data.get(session_id, {})
    
    # Extract the new messages from this turn
    new_messages = result["messages"][len(previous_messages):]
    search_result = _first_search_result_from_tool_messages(new_messages)

    # If a new HCP is found this turn that is different from the draft, reset the draft
    if search_result and existing_draft.get("hcp_id") and search_result.get("id") != existing_draft.get("hcp_id"):
        existing_draft = {}

    form_data = {
        "hcp_id": search_result.get("id") if search_result else existing_draft.get("hcp_id"),
        "hcp_name": search_result.get("name") if search_result else existing_draft.get("hcp_name"),
        "interaction_type": existing_draft.get("interaction_type"),
        "interaction_date": existing_draft.get("interaction_date"),
        "interaction_time": existing_draft.get("interaction_time"),
        "attendees": existing_draft.get("attendees"),
        "topics_discussed": existing_draft.get("topics_discussed"),
        "materials_shared": existing_draft.get("materials_shared") or [],
        "samples_distributed": existing_draft.get("samples_distributed") or [],
        "hcp_sentiment": existing_draft.get("hcp_sentiment"),
        "outcomes": existing_draft.get("outcomes"),
        "follow_up_actions": existing_draft.get("follow_up_actions"),
        "compliance_verified": existing_draft.get("compliance_verified") or False,
    }

    # Parse from user message
    msg_lower = message.lower()

    # Extract HCP name (more flexible patterns)
    hcp_match = re.search(
        r"(?:Dr\.?|Doctor)\s+([A-Za-z]+)\s+([A-Za-z]+)", message, re.IGNORECASE
    )
    if hcp_match:
        # If user explicitly provided a name and it's different from search result or draft, we might need a new search
        # But for now, just update the name and try to find ID if not already matched this turn
        new_name = f"Dr. {hcp_match.group(1)} {hcp_match.group(2)}"
        if new_name != form_data["hcp_name"]:
            form_data["hcp_name"] = new_name
            # If we don't have a search result this turn, try a fallback lookup for this new name
            if not search_result:
                fallback_hcp = _find_hcp_by_name(new_name)
                if fallback_hcp:
                    form_data["hcp_id"] = fallback_hcp.get("id")
                    form_data["hcp_name"] = fallback_hcp.get("name")
                else:
                    form_data["hcp_id"] = None
    else:
        meet_match = re.search(
            r"(?:met with|met|spoke with|visited|called|chatted with)\s+(?:Dr\.?\s*)?([A-Za-z]+)\s+([A-Za-z]+)",
            message,
            re.IGNORECASE,
        )
        if meet_match:
            new_name = f"Dr. {meet_match.group(1)} {meet_match.group(2)}"
            if new_name != form_data["hcp_name"]:
                form_data["hcp_name"] = new_name
                if not search_result:
                    fallback_hcp = _find_hcp_by_name(new_name)
                    if fallback_hcp:
                        form_data["hcp_id"] = fallback_hcp.get("id")
                        form_data["hcp_name"] = fallback_hcp.get("name")
                    else:
                        form_data["hcp_id"] = None
        else:
            simple_match = re.search(r"(?:Dr\.)\s*([A-Za-z]+)", message, re.IGNORECASE)
            if simple_match and not form_data["hcp_name"]:
                form_data["hcp_name"] = f"Dr. {simple_match.group(1)}"

    # Extract interaction type
    interaction_type_patterns = [
        (r"\bvideo call\b", "Video"),
        (r"\bvideo conference\b", "Video"),
        (r"\bzoom call\b", "Video"),
        (r"\b face to face \b", "Meeting"),
        (r"\bin-person meeting\b", "Meeting"),
        (r"\bin person meeting\b", "Meeting"),
        (r"\bmeeting\b", "Meeting"),
        (r"\bmet\b", "Meeting"),
        (r"\bvisited\b", "Meeting"),
        (r"\bphone call\b", "Call"),
        (r"\btelephone call\b", "Call"),
        (r"\bcall\b", "Call"),
        (r"\bemail\b", "Email"),
    ]
    for pattern, itype in interaction_type_patterns:
        if re.search(pattern, msg_lower):
            form_data["interaction_type"] = itype
            break

    new_date = _extract_date(message)
    if new_date:
        form_data["interaction_date"] = new_date

    time_match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", message, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or "0")
        if time_match.group(3).lower() == "pm" and hour != 12:
            hour += 12
        if time_match.group(3).lower() == "am" and hour == 12:
            hour = 0
        form_data["interaction_time"] = f"{hour:02d}:{minute:02d}"
    else:
        time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", message)
        if time_match:
            form_data["interaction_time"] = f"{int(time_match.group(1)):02d}:{time_match.group(2)}"

    # Extract sentiment
    sentiment_patterns = [
        (r"\bvery positive\b", "Positive"),
        (r"\breally positive\b", "Positive"),
        (r"\bextremely positive\b", "Positive"),
        (r"\bpositive\b", "Positive"),
        (r"\benthusiastic\b", "Positive"),
        (r"\beager\b", "Positive"),
        (r"\binterested\b", "Positive"),
        (r"\bexcited\b", "Positive"),
        (r"\bvery negative\b", "Negative"),
        (r"\breally negative\b", "Negative"),
        (r"\bextremely negative\b", "Negative"),
        (r"\bnegative\b", "Negative"),
        (r"\bresistant\b", "Negative"),
        (r"\bunhappy\b", "Negative"),
        (r"\bdisappointed\b", "Negative"),
    ]
    for pattern, sentiment in sentiment_patterns:
        if re.search(pattern, msg_lower):
            form_data["hcp_sentiment"] = sentiment
            break

    if not form_data["hcp_sentiment"]:
        form_data["hcp_sentiment"] = "Neutral"

    # Extract topics
    topics_patterns = [
        r"discussed\s+(.+?)(?:,|\s+sentiment|\s+outcome|\s+follow(?:-|\s*)up|\.|\?|$)",
        r"talked about\s+(.+?)(?:,|\s+sentiment|\s+outcome|\s+follow(?:-|\s*)up|\.|\?|$)",
        r"mentioned\s+(.+?)(?:,|\s+sentiment|\.|\?|$)",
        r"reviewed\s+(.+?)(?:,|\s+sentiment|\.|\?|$)",
    ]
    for pattern in topics_patterns:
        topics_match = re.search(pattern, message, re.IGNORECASE)
        if topics_match:
            form_data["topics_discussed"] = topics_match.group(1).strip()
            break

    if not form_data["topics_discussed"]:
        product_patterns = [
            r"product\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"drug\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:new |novel )?therapy\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"treatment\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        ]
        for pattern in product_patterns:
            product_match = re.search(pattern, message, re.IGNORECASE)
            if product_match:
                form_data["topics_discussed"] = product_match.group(1).strip()
                break

    attendees_match = re.search(r"(?:attendees?|with)\s*:\s*(.+?)(?:\.|$)", message, re.IGNORECASE)
    if attendees_match:
        form_data["attendees"] = [
            part.strip()
            for part in re.split(r",|\band\b", attendees_match.group(1))
            if part.strip()
        ]

    materials = _extract_list_after_keywords(
        message,
        (
            "shared",
            "provided",
            "sent",
            "gave",
        ),
    )
    if materials:
        form_data["materials_shared"] = [{"name": material} for material in materials]

    samples = _extract_list_after_keywords(
        message,
        (
            "sampled",
            "samples?",
            "distributed",
        ),
    )
    if samples:
        form_data["samples_distributed"] = [
            {"name": sample, "quantity": 1} for sample in samples
        ]

    outcome_match = re.search(
        r"(?:outcome(?:s)?|result(?:s)?|agreed(?: to)?|decided(?: to)?)[:\s]+(.+?)(?:,?\s+follow(?:-|\s*)up|\.|$)",
        message,
        re.IGNORECASE,
    )
    if outcome_match:
        form_data["outcomes"] = outcome_match.group(1).strip()

    follow_up_match = re.search(
        r"follow(?:-|\s*)up(?: actions?)?[:\s]+(.+?)(?:\.|$)",
        message,
        re.IGNORECASE,
    )
    if follow_up_match:
        form_data["follow_up_actions"] = follow_up_match.group(1).strip()

    logged_this_turn = any(tc.get("tool") == "log_interaction" for tc in tool_calls_made)
    if not logged_this_turn and _has_draft_data(form_data):
        _pending_form_data[session_id] = dict(form_data)
        response_text = _build_confirmation_response(form_data)

    # Update session history with all messages from this interaction
    _session_history[session_id] = list(result["messages"])

    return {
        "response": response_text,
        "tool_calls": tool_calls_made,
        "session_id": session_id,
        "form_data": form_data,
    }

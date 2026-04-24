from fastapi import APIRouter
from app.schemas.interaction import ChatRequest, ChatResponse
from app.agent.graph import invoke_agent

router = APIRouter(prefix="/api/v1/chat", tags=["AI Chat"])


@router.post("/", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    """Send a message to the LangGraph CRM agent.

    The agent extracts details, populates form, asks for confirmation, then logs.
    """
    result = invoke_agent(
        message=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(
        response=result["response"],
        tool_calls=result.get("tool_calls", []),
        form_data=result.get("form_data"),
        session_id=result["session_id"],
    )

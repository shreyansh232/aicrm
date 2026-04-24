from pydantic import BaseModel
from typing import Optional
from datetime import date, time, datetime


class InteractionBase(BaseModel):
    hcp_id: int
    interaction_type: str  # Meeting, Call, Video, Email
    interaction_date: date
    interaction_time: Optional[time] = None
    attendees: list[str] = []
    topics_discussed: Optional[str] = None
    voice_note_summary: Optional[str] = None
    materials_shared: list[dict] = []  # [{name, type}]
    samples_distributed: list[dict] = []  # [{name, quantity, lot_number}]
    hcp_sentiment: str = "Neutral"  # Positive, Neutral, Negative
    outcomes: Optional[str] = None
    follow_up_actions: Optional[str] = None
    rep_name: Optional[str] = None
    compliance_verified: bool = False


class InteractionCreate(InteractionBase):
    pass


class InteractionUpdate(BaseModel):
    hcp_id: Optional[int] = None
    interaction_type: Optional[str] = None
    interaction_date: Optional[date] = None
    interaction_time: Optional[time] = None
    attendees: Optional[list[str]] = None
    topics_discussed: Optional[str] = None
    voice_note_summary: Optional[str] = None
    materials_shared: Optional[list[dict]] = None
    samples_distributed: Optional[list[dict]] = None
    hcp_sentiment: Optional[str] = None
    outcomes: Optional[str] = None
    follow_up_actions: Optional[str] = None
    rep_name: Optional[str] = None
    compliance_verified: Optional[bool] = None


class InteractionResponse(InteractionBase):
    id: int
    ai_summary: Optional[str] = None
    ai_extracted_entities: Optional[dict] = None
    ai_suggested_follow_ups: list[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict] = []
    form_data: Optional[dict] = None
    session_id: str

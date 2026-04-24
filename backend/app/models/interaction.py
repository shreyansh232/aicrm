import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Date,
    Time,
)
from sqlalchemy.dialects.postgresql import JSON
from database import Base


class Interaction(Base):
    """Logged HCP interaction — matches the UI form fields exactly."""

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    hcp_id = Column(Integer, ForeignKey("hcps.id"), nullable=False, index=True)

    # Core fields shown in UI
    interaction_type = Column(String(50), nullable=False)  # Meeting, Call, Video, Email
    interaction_date = Column(Date, nullable=False)
    interaction_time = Column(Time, nullable=True)
    attendees = Column(JSON, default=list)  # list of attendee names/ids

    # Topics & content
    topics_discussed = Column(Text, nullable=True)  # free-text discussion points
    voice_note_summary = Column(Text, nullable=True)  # AI-generated from voice note

    # Materials & Samples
    materials_shared = Column(JSON, default=list)  # [{name, type}]
    samples_distributed = Column(JSON, default=list)  # [{name, quantity, lot_number}]

    # Sentiment & Outcomes
    hcp_sentiment = Column(String(20), default="Neutral")  # Positive, Neutral, Negative
    outcomes = Column(Text, nullable=True)  # key outcomes or agreements
    follow_up_actions = Column(Text, nullable=True)  # next steps or tasks

    # AI-generated fields
    ai_summary = Column(Text, nullable=True)
    ai_extracted_entities = Column(JSON, nullable=True)  # products, topics, etc.
    ai_suggested_follow_ups = Column(JSON, default=list)  # list of AI suggestions

    # Meta
    rep_name = Column(String(150), nullable=True)
    compliance_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

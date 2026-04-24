import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.database import Base


class HCP(Base):
    """Healthcare Professional model."""

    __tablename__ = "hcps"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    specialty = Column(String(150), nullable=True)
    institution = Column(String(250), nullable=True)
    npi_number = Column(String(20), unique=True, nullable=True)
    email = Column(String(200), nullable=True)
    phone = Column(String(30), nullable=True)
    territory = Column(String(100), nullable=True)
    tier = Column(String(1), nullable=True)  # A, B, or C (KOL tier)
    preferred_channel = Column(String(50), nullable=True)
    consent_status = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class HCPBase(BaseModel):
    first_name: str
    last_name: str
    specialty: Optional[str] = None
    institution: Optional[str] = None
    npi_number: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    territory: Optional[str] = None
    tier: Optional[str] = None
    preferred_channel: Optional[str] = None
    consent_status: bool = False


class HCPCreate(HCPBase):
    pass


class HCPUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    specialty: Optional[str] = None
    institution: Optional[str] = None
    npi_number: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    territory: Optional[str] = None
    tier: Optional[str] = None
    preferred_channel: Optional[str] = None
    consent_status: Optional[bool] = None


class HCPResponse(HCPBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

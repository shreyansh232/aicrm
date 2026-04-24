from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.interaction import Interaction
from app.schemas.interaction import (
    InteractionCreate,
    InteractionUpdate,
    InteractionResponse,
)

router = APIRouter(prefix="/api/v1/interactions", tags=["Interactions"])


@router.get("/", response_model=list[InteractionResponse])
def list_interactions(
    hcp_id: int = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List interactions, optionally filtered by HCP."""
    query = db.query(Interaction)
    if hcp_id:
        query = query.filter(Interaction.hcp_id == hcp_id)
    return query.order_by(Interaction.interaction_date.desc()).limit(limit).all()


@router.get("/{interaction_id}", response_model=InteractionResponse)
def get_interaction(interaction_id: int, db: Session = Depends(get_db)):
    """Get a single interaction by ID."""
    record = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interaction not found")
    return record


@router.post("/", response_model=InteractionResponse, status_code=201)
def create_interaction(data: InteractionCreate, db: Session = Depends(get_db)):
    """Create a new interaction record (structured form submission)."""
    record = Interaction(**data.model_dump())
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.put("/{interaction_id}", response_model=InteractionResponse)
def update_interaction(
    interaction_id: int,
    data: InteractionUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing interaction."""
    record = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interaction not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(record, field, value)
    db.commit()
    db.refresh(record)
    return record


@router.delete("/{interaction_id}", status_code=204)
def delete_interaction(interaction_id: int, db: Session = Depends(get_db)):
    """Delete an interaction."""
    record = db.query(Interaction).filter(Interaction.id == interaction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Interaction not found")
    db.delete(record)
    db.commit()

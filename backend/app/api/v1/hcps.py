from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.hcp import HCP
from app.schemas.hcp import HCPCreate, HCPUpdate, HCPResponse

router = APIRouter(prefix="/api/v1/hcps", tags=["HCPs"])


@router.get("/", response_model=list[HCPResponse])
def list_hcps(search: str = "", db: Session = Depends(get_db)):
    """List all HCPs, optionally filtered by search term."""
    query = db.query(HCP)
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            (HCP.first_name.ilike(pattern))
            | (HCP.last_name.ilike(pattern))
            | (HCP.specialty.ilike(pattern))
            | (HCP.institution.ilike(pattern))
            | (HCP.npi_number.ilike(pattern))
        )
    return query.order_by(HCP.last_name).all()


@router.get("/{hcp_id}", response_model=HCPResponse)
def get_hcp(hcp_id: int, db: Session = Depends(get_db)):
    """Get a single HCP by ID."""
    hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
    if not hcp:
        raise HTTPException(status_code=404, detail="HCP not found")
    return hcp


@router.post("/", response_model=HCPResponse, status_code=201)
def create_hcp(data: HCPCreate, db: Session = Depends(get_db)):
    """Create a new HCP record."""
    hcp = HCP(**data.model_dump())
    db.add(hcp)
    db.commit()
    db.refresh(hcp)
    return hcp


@router.put("/{hcp_id}", response_model=HCPResponse)
def update_hcp(hcp_id: int, data: HCPUpdate, db: Session = Depends(get_db)):
    """Update an existing HCP."""
    hcp = db.query(HCP).filter(HCP.id == hcp_id).first()
    if not hcp:
        raise HTTPException(status_code=404, detail="HCP not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(hcp, field, value)
    db.commit()
    db.refresh(hcp)
    return hcp

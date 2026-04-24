"""
Seed script — populates the database with sample HCP records for development.
Run with: python -m app.seed
"""

from app.database import SessionLocal
from app.models.hcp import HCP


SAMPLE_HCPS = [
    {
        "first_name": "Rajesh",
        "last_name": "Sharma",
        "specialty": "Cardiology",
        "institution": "AIIMS Delhi",
        "npi_number": "NPI1234567890",
        "email": "r.sharma@aiims.edu",
        "phone": "+91-9876543210",
        "territory": "North India",
        "tier": "A",
        "preferred_channel": "In-Person",
        "consent_status": True,
    },
    {
        "first_name": "Priya",
        "last_name": "Patel",
        "specialty": "Oncology",
        "institution": "Tata Memorial Hospital",
        "npi_number": "NPI2345678901",
        "email": "p.patel@tmc.gov.in",
        "phone": "+91-9876543211",
        "territory": "West India",
        "tier": "A",
        "preferred_channel": "Email",
        "consent_status": True,
    },
    {
        "first_name": "Arun",
        "last_name": "Kumar",
        "specialty": "Endocrinology",
        "institution": "Manipal Hospital Bangalore",
        "npi_number": "NPI3456789012",
        "email": "a.kumar@manipal.edu",
        "phone": "+91-9876543212",
        "territory": "South India",
        "tier": "B",
        "preferred_channel": "Video Call",
        "consent_status": True,
    },
    {
        "first_name": "Sunita",
        "last_name": "Reddy",
        "specialty": "Neurology",
        "institution": "Apollo Hospitals Hyderabad",
        "npi_number": "NPI4567890123",
        "email": "s.reddy@apollo.com",
        "phone": "+91-9876543213",
        "territory": "South India",
        "tier": "B",
        "preferred_channel": "In-Person",
        "consent_status": False,
    },
    {
        "first_name": "Vikram",
        "last_name": "Singh",
        "specialty": "Pulmonology",
        "institution": "Fortis Hospital Gurgaon",
        "npi_number": "NPI5678901234",
        "email": "v.singh@fortis.com",
        "phone": "+91-9876543214",
        "territory": "North India",
        "tier": "C",
        "preferred_channel": "Phone",
        "consent_status": True,
    },
    {
        "first_name": "Meera",
        "last_name": "Joshi",
        "specialty": "Rheumatology",
        "institution": "KEM Hospital Mumbai",
        "npi_number": "NPI6789012345",
        "email": "m.joshi@kem.edu",
        "phone": "+91-9876543215",
        "territory": "West India",
        "tier": "A",
        "preferred_channel": "In-Person",
        "consent_status": True,
    },
]


def seed_hcps():
    db = SessionLocal()
    try:
        existing_count = db.query(HCP).count()
        if existing_count > 0:
            print(f"Database already has {existing_count} HCPs. Skipping seed.")
            return

        for hcp_data in SAMPLE_HCPS:
            db.add(HCP(**hcp_data))
        db.commit()
        print(f"✅ Seeded {len(SAMPLE_HCPS)} sample HCPs successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_hcps()

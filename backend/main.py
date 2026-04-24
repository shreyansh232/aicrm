"""
AI-First CRM — FastAPI Application Entry Point

Serves the HCP interaction management APIs and the LangGraph AI chat agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.api.v1 import hcps, interactions, chat

settings = get_settings()

app = FastAPI(
    title="AI-First CRM — HCP Module",
    description="CRM system for pharmaceutical field representatives to manage HCP interactions with AI assistance.",
    version="1.0.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(hcps.router)
app.include_router(interactions.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {"message": "AI-First CRM HCP Module API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}

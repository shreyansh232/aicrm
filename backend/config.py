from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ai_crm"
    GROQ_API_KEY: str = "your_groq_api_key_here"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

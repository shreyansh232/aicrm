from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ai_crm"
    GROQ_API_KEY: str = "your_groq_api_key_here"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str = ""  # Add your OpenAI key here for fallback
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()

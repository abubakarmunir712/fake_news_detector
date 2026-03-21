import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    secret_key: str
    database: str
    gemini_api_key: str | None
    tavily_api_key: str | None
    port: int
    timeout: int
    cors_origins: List[str]

    @classmethod
    def load(cls) -> "Config":
        frontend_origin = os.getenv("FRONTEND_ORIGIN")
        cors_origins = (
            [frontend_origin]
            if frontend_origin
            else [
                "http://localhost:8000",
                "http://127.0.0.1:8000",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            ]
        )

        return cls(
            secret_key=os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me"),
            database=os.getenv("DATABASE_URL"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            port=int(os.getenv("PORT", 8000)),
            timeout=int(os.getenv("TIMEOUT", 8)),
            cors_origins=cors_origins,
        )

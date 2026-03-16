"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    session_secret: str = ""
    mongo_host: str = "localhost"
    mongo_port: int = 27018
    mongo_database: str = "screenapp"
    mongo_dashboard_database: str = "asr_correction_dashboard"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def mongo_uri(self) -> str:
        return f"mongodb://{self.mongo_host}:{self.mongo_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()

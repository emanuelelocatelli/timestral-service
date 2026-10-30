import secrets
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: int = 8003
    host: str = "0.0.0.0"
    model_name: str = "mistralai/Timestral-209M"
    device: str = "cpu"
    torch_num_threads: int = 4
    default_prediction_horizon: int = 8
    default_interval_seconds: int = 3600
    max_batch_series: int = 64
    service_version: str = "1.1.0"
    log_level: str = "INFO"
    api_token: str = ""
    database_path: str = "data/timestral.db"
    admin_email: str = "emanuele.locatelli@hotmail.it"
    admin_password: str = "change-me"
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))
    trusted_proxies: str = "172.18.0.2/32"


@lru_cache
def get_settings() -> Settings:
    return Settings()

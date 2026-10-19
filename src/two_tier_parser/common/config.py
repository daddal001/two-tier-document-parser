from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Fast Parser
    FAST_PARSER_HOST: str = Field("0.0.0.0", env="FAST_PARSER_HOST")
    FAST_PARSER_PORT: int = Field(8004, env="FAST_PARSER_PORT")
    WORKERS: int = Field(4, env="WORKERS")
    PYTHON_GIL: int = Field(0, env="PYTHON_GIL")

    # Accurate Parser
    ACCURATE_PARSER_HOST: str = Field("0.0.0.0", env="ACCURATE_PARSER_HOST")
    ACCURATE_PARSER_PORT: int = Field(8005, env="ACCURATE_PARSER_PORT")
    UVICORN_TIMEOUT_KEEP_ALIVE: int = Field(600, env="UVICORN_TIMEOUT_KEEP_ALIVE")

    # Logging
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

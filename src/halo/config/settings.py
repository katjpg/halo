import logging
from dataclasses import dataclass, field
from os import getenv
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from halo.core.errors import ConfigurationError
from halo.types import AgentRole, ReasoningEffort

ENV_FILE = ".env"
CONFIG_FILE = Path("configs/config.yaml")
DEFAULT_MAX_TOKENS = 1800
logger = logging.getLogger(__name__)


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENAI_", env_file=ENV_FILE, extra="ignore"
    )

    api_key: SecretStr | None = None
    base_url: str | None = None
    max_retries: int = 2
    timeout: float = 180.0

    def require_api_key(self) -> str:
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            raise ConfigurationError("OPENAI_API_KEY is not set.")
        return self.api_key.get_secret_value()


class SupabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SUPABASE_", env_file=ENV_FILE, extra="ignore"
    )

    url: str | None = None
    publishable_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    auth_issuer: str | None = None


class ModelSettings(BaseModel, extra="forbid", frozen=True):
    model: str
    max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, gt=0)
    reasoning_effort: ReasoningEffort = "low"
    temperature: float | None = None

    @model_validator(mode="after")
    def check_temperature_is_supported(self) -> "ModelSettings":
        if self.temperature is not None and self.reasoning_effort != "none":
            raise ValueError("Set reasoning_effort to 'none' when using temperature.")
        return self


class OptimizerSettings(ModelSettings):
    model: str = "gpt-6.1-terra"
    reasoning_effort: ReasoningEffort = "none"
    samples: int = Field(default=5, gt=0)


class ServerSettings(BaseModel, extra="forbid"):
    log_level: Literal[
        "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL", "NOTSET"
    ] = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


class AgentSettings(ModelSettings):
    model: str = "gpt-6-luna"
    roles: dict[AgentRole, ModelSettings] = Field(default_factory=dict)


class RuntimeSettings(BaseModel, extra="forbid"):
    max_model_calls: int = Field(default=5, gt=0)
    max_context_bytes: int = Field(default=12_000, gt=0)
    capture_prompts: bool = False


class ModelRates(BaseModel, extra="forbid", frozen=True):
    """Standard short-context rates in USD per million tokens."""

    input: float = Field(ge=0, allow_inf_nan=False)
    cached_input: float = Field(ge=0, allow_inf_nan=False)
    output: float = Field(ge=0, allow_inf_nan=False)


def default_model_rates() -> dict[str, ModelRates]:
    return {
        "gpt-6.1-terra": ModelRates(input=1.00, cached_input=0.05, output=5.00),
        "gpt-6-luna": ModelRates(input=0.10, cached_input=0.01, output=0.50),
    }


class IntegrationSettings(BaseModel, extra="forbid"):
    """Service request rates and the PyTDC cache path.

    Resolve relative paths from the working directory at startup.
    """

    chembl_requests_per_second: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    pubchem_requests_per_second: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    europe_pmc_requests_per_second: float = Field(
        default=2.0, gt=0, allow_inf_nan=False
    )
    semantic_scholar_requests_per_second: float = Field(
        default=1.0, gt=0, allow_inf_nan=False
    )
    oracle_data_dir: Path = Path("oracle")


DEFAULT_INTEGRATIONS = IntegrationSettings()


class AppConfig(BaseModel, extra="forbid"):
    optimizer: OptimizerSettings = OptimizerSettings()
    server: ServerSettings = ServerSettings()
    agents: AgentSettings = AgentSettings()
    runtime: RuntimeSettings = RuntimeSettings()
    model_rates: dict[str, ModelRates] = Field(default_factory=default_model_rates)
    integrations: IntegrationSettings = DEFAULT_INTEGRATIONS


@dataclass
class Settings:
    openai: OpenAISettings
    supabase: SupabaseSettings
    optimizer: OptimizerSettings
    server: ServerSettings
    agents: AgentSettings = field(default_factory=AgentSettings)
    config_path: Path | None = None
    integrations: IntegrationSettings = field(default_factory=IntegrationSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    model_rates: dict[str, ModelRates] = field(default_factory=default_model_rates)


def load_app_config(path: Path | None = None) -> AppConfig:
    """Read YAML or warn and use defaults if the default file is missing."""
    if path is None and not CONFIG_FILE.exists():
        logger.warning("Configuration file %s is missing. Using defaults.", CONFIG_FILE)
        return AppConfig()
    path = path or CONFIG_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"Cannot read configuration file {path}.") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return AppConfig.model_validate({} if data is None else data)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} is not a valid configuration: {exc}") from exc


def load_settings() -> Settings:
    """Load credentials and resolve the optional HALO_CONFIG_PATH override."""
    override = getenv("HALO_CONFIG_PATH")
    path = Path(override).expanduser().resolve() if override is not None else None
    config = load_app_config(path)
    return Settings(
        openai=OpenAISettings(),
        supabase=SupabaseSettings(),
        optimizer=config.optimizer,
        server=config.server,
        agents=config.agents,
        config_path=path or (CONFIG_FILE.resolve() if CONFIG_FILE.is_file() else None),
        integrations=config.integrations,
        runtime=config.runtime,
        model_rates=config.model_rates,
    )

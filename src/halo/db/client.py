from collections.abc import Awaitable
from typing import TypeVar

from httpx import AsyncClient, InvalidURL, URL
from postgrest import AsyncPostgrestClient
from pydantic import SecretStr
from supabase import PostgrestAPIError
from supabase_auth import AsyncGoTrueClient

from halo.config.settings import SupabaseSettings
from halo.core.errors import (
    ConfigurationError,
    ConflictError,
    DataError,
    NotFoundError,
)

T = TypeVar("T")


async def execute(operation: Awaitable[T]) -> T:
    try:
        return await operation
    except PostgrestAPIError as exc:
        if exc.code == "PT404":
            raise NotFoundError(exc.message) from exc
        if exc.code == "PT409":
            raise ConflictError(exc.message) from exc
        raise DataError("The database request failed.") from exc


def auth_url(settings: SupabaseSettings) -> str:
    return f"{_base_url(settings)}/auth/v1"


def auth_client(settings: SupabaseSettings, http: AsyncClient) -> AsyncGoTrueClient:
    key = _key(settings.publishable_key, "SUPABASE_PUBLISHABLE_KEY")
    return AsyncGoTrueClient(
        url=auth_url(settings),
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        auto_refresh_token=False,
        persist_session=False,
        http_client=http,
    )


def service_client(
    settings: SupabaseSettings, http: AsyncClient
) -> AsyncPostgrestClient:
    key = _key(settings.secret_key, "SUPABASE_SECRET_KEY")
    return _data_client(settings, http, key, key)


def user_client(
    settings: SupabaseSettings, http: AsyncClient, token: str
) -> AsyncPostgrestClient:
    key = _key(settings.publishable_key, "SUPABASE_PUBLISHABLE_KEY")
    return _data_client(settings, http, key, token)


def _data_client(
    settings: SupabaseSettings, http: AsyncClient, key: str, token: str
) -> AsyncPostgrestClient:
    return AsyncPostgrestClient(
        f"{_base_url(settings)}/rest/v1",
        headers={"apikey": key, "Authorization": f"Bearer {token}"},
        http_client=http,
    )


def _base_url(settings: SupabaseSettings) -> str:
    if not settings.url:
        raise ConfigurationError("SUPABASE_URL is not set.")
    try:
        url = URL(settings.url)
    except InvalidURL as exc:
        raise ConfigurationError("SUPABASE_URL must be an HTTP base URL.") from exc
    if (
        url.scheme not in ("http", "https")
        or not url.host
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        raise ConfigurationError("SUPABASE_URL must be an HTTP base URL.")
    return str(url).rstrip("/")


def _key(key: SecretStr | None, name: str) -> str:
    if key is None or not key.get_secret_value().strip():
        raise ConfigurationError(f"{name} is not set.")
    return key.get_secret_value()

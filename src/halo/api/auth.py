import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError
from supabase_auth import AsyncGoTrueClient
from supabase_auth.errors import (
    AuthApiError,
    AuthError,
    AuthRetryableError,
    AuthUnknownError,
)

from halo.core.errors import AuthenticationError, ProviderError
from halo.types import UserId

BEARER_SCHEME = HTTPBearer(auto_error=False)
AUTH_TIMEOUT_S = 5


def access_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(BEARER_SCHEME)
    ],
) -> str:
    if credentials is None:
        raise AuthenticationError("A bearer token is required.")
    return credentials.credentials


async def verify_token(
    auth_client: AsyncGoTrueClient, token: str, *, issuer: str
) -> UserId:
    try:
        async with asyncio.timeout(AUTH_TIMEOUT_S):
            verified = await auth_client.get_claims(token)
    except (AuthError, ValidationError, TimeoutError) as exc:
        outage = isinstance(
            exc, (AuthRetryableError, AuthUnknownError, ValidationError, TimeoutError)
        )
        if (
            outage
            or isinstance(exc, AuthApiError)
            and (exc.status >= 500 or exc.status in (408, 429))
        ):
            raise ProviderError("Supabase Auth is unavailable.") from exc
        raise AuthenticationError("The access token is invalid.") from exc

    claims = verified["claims"] if verified else {}
    audience = claims.get("aud")
    if (
        claims.get("iss") != issuer
        or claims.get("role") != "authenticated"
        or not (
            audience == "authenticated"
            or isinstance(audience, list)
            and "authenticated" in audience
        )
    ):
        raise AuthenticationError("The access token is invalid.")
    try:
        return UserId(str(UUID(claims.get("sub", ""))))
    except ValueError as exc:
        raise AuthenticationError("The access token is invalid.") from exc

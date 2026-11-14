from typing import ClassVar

from halo.types import ApplicationErrorCode


class ApplicationError(Exception):
    """An application error with a stable client-facing code."""

    code: ClassVar[ApplicationErrorCode] = "internal_error"


class ConfigurationError(ApplicationError):
    code = "configuration_error"


class ParseError(ApplicationError, ValueError):
    code = "parse_error"


class DataError(ApplicationError):
    """A database request failed or returned invalid data."""


class AuthenticationError(ApplicationError):
    code = "unauthenticated"


class ProviderError(ApplicationError):
    code = "provider_error"


class LimitError(ApplicationError):
    code = "limit_exceeded"


class NotFoundError(ApplicationError):
    code = "not_found"


class ConflictError(ApplicationError):
    code = "conflict"

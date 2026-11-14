import logging
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from halo.core.errors import (
    ApplicationError,
    AuthenticationError,
    ConflictError,
    LimitError,
    NotFoundError,
    ParseError,
    ProviderError,
)

logger = logging.getLogger(__name__)

HTTP_STATUS_BY_ERROR = {
    ParseError: HTTPStatus.BAD_REQUEST,
    AuthenticationError: HTTPStatus.UNAUTHORIZED,
    NotFoundError: HTTPStatus.NOT_FOUND,
    ConflictError: HTTPStatus.CONFLICT,
    LimitError: HTTPStatus.UNPROCESSABLE_ENTITY,
    ProviderError: HTTPStatus.BAD_GATEWAY,
}
SERVER_FAULT_MESSAGE = "The server could not complete the request."


class ErrorResponse(BaseModel):
    code: str
    message: str


def error_body(request: Request, exc: Exception) -> tuple[int, ErrorResponse]:
    """Map a failure to its public status and message."""
    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
        first = errors[0] if errors else {}
        location = ".".join(map(str, first.get("loc", ())))
        message = first.get("msg", "Request validation failed.")
        if location:
            message = f"{location}: {message}"
        status, code = 422, "validation_error"
    elif isinstance(exc, HTTPException):
        status = exc.status_code
        try:
            code = HTTPStatus(status).phrase.lower().replace(" ", "_")
        except ValueError:
            code = "http_error"
        message = str(exc.detail)
    elif isinstance(exc, ApplicationError):
        status = next(
            (
                HTTP_STATUS_BY_ERROR[cls]
                for cls in type(exc).__mro__
                if cls in HTTP_STATUS_BY_ERROR
            ),
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        code, message = exc.code, str(exc)
    else:
        status, code, message = 500, "internal_error", str(exc)
    server_fault = status >= 500
    logger.log(
        logging.ERROR if server_fault else logging.INFO,
        "%s %s: %s: %s",
        request.method,
        request.url.path,
        code,
        message,
        exc_info=exc if server_fault else False,
    )
    return status, ErrorResponse(
        code=code, message=SERVER_FAULT_MESSAGE if server_fault else message
    )


async def handle_error(request: Request, exc: Exception) -> JSONResponse:
    status, body = error_body(request, exc)
    headers = dict(exc.headers or {}) if isinstance(exc, HTTPException) else {}
    if status == 401:
        headers.setdefault("WWW-Authenticate", "Bearer")
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


class ErrorBoundary:
    """Return masked errors before response headers have been sent."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def forward(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, forward)
        except Exception as exc:
            if started:
                raise
            response = await handle_error(Request(scope), exc)
            await response(scope, receive, send)


def register_error_handlers(app: FastAPI) -> None:
    """Hide server details and preserve public error codes."""
    for error in (Exception, ApplicationError, RequestValidationError, HTTPException):
        app.add_exception_handler(error, handle_error)

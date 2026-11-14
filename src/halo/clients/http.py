import asyncio
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Self
from urllib.parse import quote, unquote, urlsplit

import httpx

from halo.core.errors import ParseError, ProviderError

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ServiceError(ProviderError):
    """A failed request to an external data service.

    Attributes:
        status_code: HTTP status when available.
        retry_after_s: Delay the service asked for, in seconds, if it sent one.
        payload: The decoded JSON error body, if the service sent one.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.payload = payload


class ResponseError(ServiceError):
    """A successful response that breaks the service's documented contract."""


class RateLimiter:
    """Request pacing for callers on one event loop."""

    def __init__(self, requests_per_second: float) -> None:
        if not math.isfinite(requests_per_second) or requests_per_second <= 0:
            raise ParseError("requests_per_second must be finite and positive")
        self._interval_s = 1.0 / requests_per_second
        self._next = 0.0

    async def wait(self) -> None:
        """Wait until a request can start and reserve its place."""
        while not self.try_start():
            await self.wait_ready()

    async def wait_ready(self) -> None:
        """Wait until a request may start."""
        while (delay := self._next - time.monotonic()) > 0:
            await asyncio.sleep(delay)

    def try_start(self) -> bool:
        """Reserve a request if the required pause has elapsed."""
        now = time.monotonic()
        if now < self._next:
            return False
        self._next = now + self._interval_s
        return True

    def defer(self, seconds: float) -> None:
        """Delay later requests without shortening an existing pause."""
        if math.isfinite(seconds) and seconds > 0:
            self._next = max(self._next, time.monotonic() + seconds)


class HttpClient:
    """A JSON and XML client with pacing, retries, and caching for one service.

    Timeouts apply per network phase. max_attempts includes the first call.
    A Retry-After beyond max_retry_wait_s ends retries for that request.
    Cache limits count response bytes and entries. Injected clients stay open.
    """

    def __init__(
        self,
        base_url: str,
        *,
        limiter: RateLimiter | None = None,
        headers: Mapping[str, str] | None = None,
        concurrency: int = 2,
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        max_retry_wait_s: float = 60.0,
        cache_ttl_s: float = 300.0,
        cache_size: int = 128,
        max_entry_bytes: int = 1_048_576,
        max_cache_bytes: int = 8 * 1024**2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        try:
            url = urlsplit(base_url)
        except ValueError as exc:
            raise ParseError("invalid service base_url") from exc
        hostname = url.hostname
        if url.scheme != "https" or not hostname or url.query or url.fragment:
            raise ParseError("base_url must be an HTTPS URL without query or fragment")
        if url.username is not None:
            raise ParseError("base_url must not include credentials")
        if concurrency < 1 or max_attempts < 1:
            raise ParseError("concurrency and max_attempts must be at least 1")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ParseError("timeout_s must be finite and positive")
        if any(not math.isfinite(s) or s < 0 for s in (max_retry_wait_s, cache_ttl_s)):
            raise ParseError(
                "max_retry_wait_s and cache_ttl_s must be finite and nonnegative"
            )
        if min(cache_size, max_entry_bytes, max_cache_bytes) < 0:
            raise ParseError("cache limits must be nonnegative")
        self._base_url = base_url.rstrip("/")
        self._service = hostname
        self.limiter = limiter or RateLimiter(2.0)
        self._slots = asyncio.Semaphore(concurrency)
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._max_retry_wait_s = max_retry_wait_s
        self._cache_ttl_s = cache_ttl_s
        self._cache_size = cache_size
        self._max_entry_bytes = max_entry_bytes
        self._max_cache_bytes = max_cache_bytes
        self._cached_bytes = 0
        self._cache: OrderedDict[bytes, tuple[float, bytes]] = OrderedDict()
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "halo-integrations/1.0",
            **(headers or {}),
        }
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close owned connections and drop cached responses."""
        self._cache.clear()
        self._cached_bytes = 0
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def request_json(
        self,
        method: str,
        path: str | tuple[str, ...],
        *,
        params: Mapping[str, str | int | float | bool] | None = None,
        data: Mapping[str, str] | None = None,
        json_body: Any = None,
        replayable: bool = False,
        cache: bool = False,
        response_hook: Callable[[httpx.Response], None] | None = None,
    ) -> Any:
        """Send a request and decode its JSON response.

        Args:
            method: HTTP method.
            path: A relative path or raw path segments to encode.
            params: Query parameters.
            data: Form fields.
            json_body: JSON request body.
            replayable: Allow retries for methods other than GET.
            cache: Reuse responses for replayable requests.
            response_hook: Callback for each network response.

        Returns:
            A fresh JSON value from an HTTP 200 response.
        """
        return await self._request(
            method,
            path,
            params=params,
            data=data,
            json_body=json_body,
            replayable=replayable,
            cache=cache,
            response_hook=response_hook,
        )

    async def request_xml(
        self,
        method: str,
        path: str | tuple[str, ...],
        *,
        params: Mapping[str, str | int | float | bool] | None = None,
        cache: bool = False,
    ) -> bytes:
        """Read XML bytes, preserving their encoding and limiting the body to 8 MiB."""
        return await self._request(
            method, path, params=params, cache=cache, response_format="xml"
        )

    async def _request(
        self,
        method: str,
        path: str | tuple[str, ...],
        *,
        params: Mapping[str, str | int | float | bool] | None = None,
        data: Mapping[str, str] | None = None,
        json_body: Any = None,
        replayable: bool = False,
        cache: bool = False,
        response_hook: Callable[[httpx.Response], None] | None = None,
        response_format: Literal["json", "xml"] = "json",
    ) -> Any:
        if self._closed or self._client.is_closed:
            raise ServiceError("HTTP client is closed")
        if isinstance(path, tuple):
            if not path or any(
                not segment
                or any(
                    part in (".", "..")
                    for part in unquote(segment).replace("\\", "/").split("/")
                )
                for segment in path
            ):
                raise ParseError("path must be relative to the service root")
            path = "/".join(quote(segment, safe="") for segment in path)
        else:
            decoded = unquote(path)
            if (
                not path
                or decoded.startswith("/")
                or any(part in (".", "..") for part in decoded.split("/"))
                or any(ch in decoded for ch in (":", "?", "#", "\\"))
            ):
                raise ParseError("path must be relative to the service root")
        method = method.upper()
        replayable = replayable or method == "GET"
        if cache and not replayable:
            raise ParseError("only replayable requests may be cached")
        request = self._client.build_request(
            method,
            f"{self._base_url}/{path}",
            params=params,
            data=data,
            json=json_body,
            headers=self._headers
            | ({"Accept": "application/xml"} if response_format == "xml" else {}),
            timeout=self._timeout_s,
        )
        # Retries reuse the request, so its body must already be buffered.
        try:
            content = request.content
        except httpx.RequestNotRead as exc:
            raise ParseError("HTTP requests need a buffered body") from exc
        key = None
        if cache:
            key = hashlib.sha256(
                b"\0".join(
                    (
                        method.encode(),
                        str(request.url).encode(),
                        content,
                        response_format.encode(),
                    )
                )
            ).digest()
            if (hit := self._cached(key)) is not None:
                return json.loads(hit) if response_format == "json" else hit

        attempts = self._max_attempts if replayable else 1
        for attempt in range(1, attempts + 1):
            backoff_s = min(
                2 ** (attempt - 1) + random.random(), self._max_retry_wait_s
            )
            try:
                response = (
                    await self._send(request, max_response_bytes=8 * 1024**2)
                    if response_format == "xml"
                    else await self._send(request)
                )
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as exc:
                if attempt == attempts:
                    raise ServiceError(f"{self._service} request failed") from exc
                self.limiter.defer(backoff_s)
                continue
            if response_hook is not None:
                response_hook(response)

            if response.status_code == 200:
                payload = response.content
                if response_format == "json":
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise ResponseError(
                            f"{self._service} returned invalid JSON"
                        ) from exc
                if key is not None:
                    self._store(key, response)
                return payload

            retry_after_s = _parse_retry_after(response.headers.get("Retry-After"))
            if retry_after_s is not None:
                self.limiter.defer(retry_after_s)
            try:
                error_body = response.json()
            except ValueError:
                error_body = None
            message = f"{self._service} returned HTTP {response.status_code}"
            if response.status_code in (301, 302, 303, 307, 308):
                message += ". Redirects are disabled"
            error = ServiceError(
                message,
                status_code=response.status_code,
                retry_after_s=retry_after_s,
                payload=error_body,
            )
            if (
                response.status_code not in RETRYABLE_STATUS
                or attempt == attempts
                or (
                    retry_after_s is not None and retry_after_s > self._max_retry_wait_s
                )
            ):
                raise error
            self.limiter.defer(max(retry_after_s or 0.0, backoff_s))
        raise AssertionError("unreachable: the last attempt returns or raises")

    async def _send(
        self, request: httpx.Request, *, max_response_bytes: int | None = None
    ) -> httpx.Response:
        while True:
            await self.limiter.wait_ready()
            async with self._slots:
                if self._closed or self._client.is_closed:
                    raise ServiceError("HTTP client is closed")
                # Pacing can change while this request waits for a slot.
                if self.limiter.try_start():
                    if max_response_bytes is None:
                        return await self._client.send(request, follow_redirects=False)
                    response = await self._client.send(
                        request, follow_redirects=False, stream=True
                    )
                    try:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(body) + len(chunk) > max_response_bytes:
                                raise ResponseError(
                                    f"{self._service} XML exceeds the response size limit"
                                )
                            body.extend(chunk)
                        headers = {
                            key: value
                            for key, value in response.headers.items()
                            if key.lower() not in {"content-encoding", "content-length"}
                        }
                        return httpx.Response(
                            response.status_code,
                            headers=headers,
                            content=bytes(body),
                            request=request,
                        )
                    finally:
                        await response.aclose()

    def _cached(self, key: bytes) -> bytes | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires, body = entry
        if expires <= time.monotonic():
            del self._cache[key]
            self._cached_bytes -= len(body)
            return None
        self._cache.move_to_end(key)
        return body

    def _store(self, key: bytes, response: httpx.Response) -> None:
        directives = response.headers.get("Cache-Control", "").lower()
        body = response.content
        if (
            self._closed
            or not self._cache_ttl_s
            or not self._cache_size
            or len(body) > min(self._max_entry_bytes, self._max_cache_bytes)
            or "no-store" in directives
            or "private" in directives
        ):
            return
        if previous := self._cache.pop(key, None):
            self._cached_bytes -= len(previous[1])
        self._cache[key] = (time.monotonic() + self._cache_ttl_s, body)
        self._cached_bytes += len(body)
        while (
            len(self._cache) > self._cache_size
            or self._cached_bytes > self._max_cache_bytes
        ):
            _, (_, evicted) = self._cache.popitem(last=False)
            self._cached_bytes -= len(evicted)


class ServiceClient:
    """A service adapter that owns or borrows its HTTP client."""

    def __init__(
        self,
        base_url: str,
        http: HttpClient | None = None,
        *,
        requests_per_second: float,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._owns_http = http is None
        self._http = (
            http
            if http is not None
            else HttpClient(
                base_url, limiter=RateLimiter(requests_per_second), headers=headers
            )
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()


def _parse_retry_after(value: str | None) -> float | None:
    """Read Retry-After as seconds or an HTTP date; None if absent or unreadable."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        seconds = (date - datetime.now(UTC)).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


def json_object(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ResponseError(f"expected a JSON object for {what}")
    return value


def records(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = payload.get(key, default)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ResponseError(f"expected a list of objects in {key!r}")
    return rows


def check_integer(
    value: int, name: str, *, minimum: int = 1, maximum: int | None = None
) -> None:
    """Require an integer in the given inclusive range."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        bounds = (
            f"from {minimum} to {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise ParseError(f"{name} must be an integer {bounds}")


def number(record: Mapping[str, Any], key: str) -> float | None:
    """Read a finite number or numeric string. Treat null and "" as missing."""
    value = record.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ResponseError(f"{key!r} is not a number: {value!r}")
    try:
        val = float(value)
    except (ValueError, OverflowError) as exc:
        raise ResponseError(f"{key!r} is not a number: {value!r}") from exc
    if not math.isfinite(val):
        raise ResponseError(f"{key!r} is not finite: {value!r}")
    return val


def integer(record: Mapping[str, Any], key: str) -> int | None:
    """Read an exact whole number or numeric string. Null and "" mean missing."""
    value = record.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ResponseError(f"{key!r} is not an integer: {value!r}")
    try:
        val = Decimal(str(value))
    except InvalidOperation as exc:
        raise ResponseError(f"{key!r} is not an integer: {value!r}") from exc
    if not val.is_finite() or val != val.to_integral_value():
        raise ResponseError(f"{key!r} is not an integer: {value!r}")
    return int(val)


def text(record: Mapping[str, Any], key: str) -> str | None:
    """Read a string, treating an empty string as missing."""
    value = record.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ResponseError(f"{key!r} is not text: {value!r}")
    return value


def flag(record: Mapping[str, Any], key: str) -> bool | None:
    """Read a JSON boolean or the integers 0 and 1."""
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ResponseError(f"{key!r} is not a boolean: {value!r}")

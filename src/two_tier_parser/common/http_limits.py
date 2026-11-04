"""Request-size limits shared by the parser services.

The limit is enforced while the body streams, before the multipart parser spools it to disk
or the handler reads it into memory.
"""

from typing import Any, Callable, Collection

from fastapi import HTTPException
from starlette.responses import JSONResponse


def route_path(scope: Any) -> str:
    """Return the request path the router sees, without the ASGI ``root_path`` prefix.

    Behind a path-prefixing proxy, servers such as uvicorn (``--root-path``) put the prefix
    in ``scope["path"]`` and the router strips it again; comparing the raw path would
    miss ``/parse`` and skip the body limit and admission control.
    """
    path = scope["path"]
    root = (scope.get("root_path") or "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root) :]
    return path.rstrip("/") or "/"


def exceeds(declared: str, limit: int) -> bool:
    """Return True when a ``Content-Length`` value is a number above the limit.

    ``str.isdigit`` also accepts characters such as superscript digits that ``int``
    rejects, so the value must be ASCII as well.
    """
    return declared.isascii() and declared.isdigit() and int(declared) > limit


class BodyTooLarge(HTTPException):
    """Raised from the request stream once the body passes its limit."""

    def __init__(self, detail: str) -> None:
        super().__init__(status_code=413, detail=detail)


class BodyLimitMiddleware:
    """Stop oversized request bodies while they stream, before anything is spooled.

    ``Content-Length`` is checked first so an honest oversized request is refused
    without reading a byte. A chunked request, or one that understates its length,
    is counted as it arrives and the read is aborted with 413 at the limit, so the
    multipart parser never writes the rest of the body to temporary storage.

    The limits are callables so a service can keep them as module settings that tests or
    operators change at run time.

    Args:
        app: The ASGI application to wrap.
        paths: Routes the limit applies to; every other request passes through.
        upload_limit: Largest multipart body in bytes (the file plus its framing).
        json_limit: Largest body of any other content type.
        detail: The 413 message.
    """

    def __init__(
        self,
        app: Any,
        *,
        paths: Collection[str],
        upload_limit: Callable[[], int],
        json_limit: Callable[[], int],
        detail: str,
    ) -> None:
        self.app = app
        self.paths = paths
        self.upload_limit = upload_limit
        self.json_limit = json_limit
        self.detail = detail

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        if route_path(scope) not in self.paths:
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope["headers"]}
        multipart = b"multipart/form-data" in headers.get(b"content-type", b"").lower()
        limit = self.upload_limit() if multipart else self.json_limit()

        declared = headers.get(b"content-length", b"").decode("latin-1")
        if exceeds(declared, limit):
            await JSONResponse(status_code=413, content={"detail": self.detail})(
                scope, receive, send
            )
            return

        received = 0

        async def counting_receive() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(self.detail)
            return message

        await self.app(scope, counting_receive, send)

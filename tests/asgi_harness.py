"""Drive an ASGI app with a request body that arrives in chunks."""

import asyncio


def post(app, path, headers, chunks, consumed, complete=False):
    """POST ``chunks`` to ``app``; append each chunk's size to ``consumed`` as it is read.

    With ``complete`` false the body never ends, so a server that keeps reading would keep
    consuming. Returns the response status and body.
    """
    sent = []
    iterator = iter(chunks)
    consumed_count = [0]

    async def receive():
        chunk = next(iterator, None)
        if chunk is None:
            return {"type": "http.disconnect"}
        consumed.append(len(chunk))
        consumed_count[0] += 1
        more = consumed_count[0] < len(chunks) or not complete
        return {"type": "http.request", "body": chunk, "more_body": more}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "server": ("test", 80),
        "client": ("test", 1),
        "scheme": "http",
    }
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body

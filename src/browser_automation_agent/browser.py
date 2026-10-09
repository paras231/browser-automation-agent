import time
import uuid
from playwright.async_api import async_playwright

_playwright = None
_browser = None
_sessions: dict[str, "_Session"] = {}

DEFAULT_SESSION_ID = "default"
MEDIA_PERMISSIONS = ["camera", "microphone"]

# Bound how many buffered devtools events we keep per session, so a
# long-running session (or a page that logs/polls constantly) can't grow
# these lists without limit.
_MAX_DEVTOOLS_ENTRIES = 300


class _Session:
    def __init__(self, context, page):
        self.context = context
        self.page = page
        self.console_logs: list[dict] = []
        self.network_entries: list[dict] = []
        self.page_errors: list[dict] = []
        # Maps a live Playwright Request object -> its entry dict, so a
        # later "response"/"requestfailed" event (which only gives us the
        # Request, not the entry we already appended) can be updated in
        # place. Keying by the Request object itself (not id()) keeps it
        # alive for exactly as long as we need it and avoids id() reuse
        # once an unrelated object is garbage collected.
        self._network_index: dict = {}


def _attach_devtools_listeners(session: "_Session"):
    """Buffer console messages, uncaught page errors, and network
    requests/responses for `session`, so devtools-style inspection tools
    can read them later without needing to be listening at the exact
    moment the event happened (the same way a real browser's devtools
    panels keep a running history while they're open)."""

    page = session.page

    def _trim(entries: list):
        while len(entries) > _MAX_DEVTOOLS_ENTRIES:
            removed = entries.pop(0)
            session._network_index.pop(removed.get("_request"), None)

    def _on_console(msg):
        session.console_logs.append({
            "level": msg.type,
            "text": msg.text,
            "location": msg.location,
            "timestamp": time.time(),
        })
        _trim(session.console_logs)

    def _on_page_error(error):
        session.page_errors.append({
            "message": str(error),
            "timestamp": time.time(),
        })
        _trim(session.page_errors)

    def _on_request(request):
        entry = {
            "url": request.url,
            "method": request.method,
            "resource_type": request.resource_type,
            "status": None,
            "ok": None,
            "failure": None,
            "timestamp": time.time(),
            "_request": request,
        }
        session.network_entries.append(entry)
        session._network_index[request] = entry
        _trim(session.network_entries)

    def _on_response(response):
        entry = session._network_index.get(response.request)
        if entry is not None:
            entry["status"] = response.status
            entry["ok"] = response.ok

    def _on_request_failed(request):
        entry = session._network_index.get(request)
        if entry is not None:
            entry["ok"] = False
            failure = request.failure
            entry["failure"] = failure.get("errorText") if isinstance(failure, dict) else failure

    page.on("console", _on_console)
    page.on("pageerror", _on_page_error)
    page.on("request", _on_request)
    page.on("response", _on_response)
    page.on("requestfailed", _on_request_failed)


def _public_network_entries(entries: list[dict]) -> list[dict]:
    """Strip the internal "_request" object reference before handing
    network entries back to a caller/tool - it's not serializable and
    is only needed internally to pair up request/response events."""

    return [{k: v for k, v in e.items() if k != "_request"} for e in entries]


async def _ensure_browser():
    global _playwright, _browser

    if _browser is None:
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(headless=False)

    return _browser


async def get_page(session_id: str = DEFAULT_SESSION_ID):
    """Return the Playwright page for `session_id`, lazily creating an
    isolated browser context + page for it on first use.

    `session_id` must be passed explicitly by the caller (e.g. via
    LangGraph's RunnableConfig) rather than tracked ambiently - tool
    calls inside a LangGraph run each execute in their own asyncio Task,
    and a Task only inherits the contextvar state of whoever created it,
    not of sibling Tasks that ran before it. Relying on an ambient
    contextvar here previously caused a brand new browser tab to be
    opened on almost every tool call.
    """

    if session_id not in _sessions:
        browser = await _ensure_browser()
        context = await browser.new_context(permissions=MEDIA_PERMISSIONS)
        page = await context.new_page()
        session = _Session(context, page)
        _attach_devtools_listeners(session)
        _sessions[session_id] = session

    return _sessions[session_id].page


def get_console_logs(session_id: str = DEFAULT_SESSION_ID) -> list[dict]:
    """Return the buffered console messages (console.log/warn/error/etc.)
    logged by `session_id`'s page since the session started or was last
    cleared."""

    session = _sessions.get(session_id)
    return list(session.console_logs) if session else []


def get_page_errors(session_id: str = DEFAULT_SESSION_ID) -> list[dict]:
    """Return the buffered uncaught JS exceptions thrown by `session_id`'s
    page since the session started or was last cleared."""

    session = _sessions.get(session_id)
    return list(session.page_errors) if session else []


def get_network_entries(session_id: str = DEFAULT_SESSION_ID) -> list[dict]:
    """Return the buffered HTTP requests (with status/failure once known)
    made by `session_id`'s page since the session started or was last
    cleared."""

    session = _sessions.get(session_id)
    if not session:
        return []

    return _public_network_entries(session.network_entries)


def clear_console_logs(session_id: str = DEFAULT_SESSION_ID):
    session = _sessions.get(session_id)
    if session:
        session.console_logs.clear()


def clear_network_entries(session_id: str = DEFAULT_SESSION_ID):
    session = _sessions.get(session_id)
    if session:
        session.network_entries.clear()
        session._network_index.clear()


def clear_page_errors(session_id: str = DEFAULT_SESSION_ID):
    session = _sessions.get(session_id)
    if session:
        session.page_errors.clear()


async def new_session() -> str:
    """Create a fresh, uniquely-keyed session (its own cookies, storage,
    and permission grants) and return its id. Camera and microphone
    access is pre-granted for every origin so the browser never shows a
    native permission prompt."""

    session_id = uuid.uuid4().hex
    await get_page(session_id)

    return session_id


async def close_session(session_id: str):
    session = _sessions.pop(session_id, None)

    if session is not None:
        await session.context.close()


async def grant_media_permissions(session_id: str = DEFAULT_SESSION_ID, url: str | None = None):
    """Explicitly (re-)grant camera/microphone access for `session_id`,
    optionally scoped to a single origin. Sessions already grant these
    permissions for every origin on creation, so this mainly exists for
    re-scoping to a specific site on request."""

    await get_page(session_id)
    context = _sessions[session_id].context

    if url:
        await context.grant_permissions(MEDIA_PERMISSIONS, origin=url)
    else:
        await context.grant_permissions(MEDIA_PERMISSIONS)

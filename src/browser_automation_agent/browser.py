import uuid
from playwright.async_api import async_playwright

_playwright = None
_browser = None
_sessions: dict[str, "_Session"] = {}

DEFAULT_SESSION_ID = "default"
MEDIA_PERMISSIONS = ["camera", "microphone"]


class _Session:
    def __init__(self, context, page):
        self.context = context
        self.page = page


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
        _sessions[session_id] = _Session(context, page)

    return _sessions[session_id].page


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

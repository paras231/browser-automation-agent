import asyncio
import functools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

try:
    from .browser import (
        get_page,
        grant_media_permissions as _grant_media_permissions,
        DEFAULT_SESSION_ID,
    )
    from . import captcha as _captcha
except ImportError:
    from browser import (
        get_page,
        grant_media_permissions as _grant_media_permissions,
        DEFAULT_SESSION_ID,
    )
    import captcha as _captcha


_LOG_PATH = Path("logs") / "tool_calls.log"


def _log_tool_call(fn):
    """Log every call to a tool: args (minus the RunnableConfig object,
    which is noisy and not useful here), the session id, how long the
    underlying Playwright call took, and what it returned - one line to
    stdout and one JSON record appended to logs/tool_calls.log.

    This replaces ad-hoc print() calls scattered in individual tools -
    every tool gets the same record for free, with no per-tool code.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        session_id = _session_id(kwargs.get("config"))
        logged_args = {k: v for k, v in kwargs.items() if k != "config"}
        started = time.monotonic()
        status = "ok"
        result = None

        try:
            result = await fn(*args, **kwargs)
            return result
        except Exception as e:
            status = "exception"
            result = {"error": str(e)}
            raise
        finally:
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)

            print(
                f"[tool] {fn.__name__} args={logged_args} "
                f"-> {status} in {elapsed_ms}ms"
            )

            try:
                _LOG_PATH.parent.mkdir(exist_ok=True)

                with open(_LOG_PATH, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "tool": fn.__name__,
                        "session_id": session_id,
                        "args": logged_args,
                        "result": result,
                        "status": status,
                        "elapsed_ms": elapsed_ms,
                    }, default=str) + "\n")
            except Exception as log_error:
                print(f"[tool] failed to write log entry: {log_error}")

    return wrapper


async def _with_retries(action, attempts: int = 3, base_delay: float = 0.4):
    """Retry a flaky Playwright step (element not visible *yet*, a page
    still animating/re-rendering, a brief navigation hiccup) before
    surfacing a failure to the agent.

    This is only for transient timing issues - it does not change outcome
    for a genuinely wrong selector or a real validation error, which will
    fail identically on every attempt and just return promptly. Keeping
    this inside the tool (instead of relying on the LLM to notice a
    failure and retry via another tool call) fixes the common flaky case
    without spending an extra agent turn on it.
    """
    last_error = None

    for attempt in range(attempts):
        try:
            return await action()
        except Exception as e:
            last_error = e

            if attempt < attempts - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))

    raise last_error


def _session_id(config: RunnableConfig | None) -> str:
    """Pull the browser session id out of the graph's RunnableConfig.

    This is threaded explicitly by LangGraph through every node and tool
    call, unlike an ambient contextvar - each tool call runs in its own
    asyncio Task spawned from the Pregel scheduler's own context, which
    does not see contextvar mutations made by a sibling Task that ran
    earlier. Using config is what keeps every tool call in one agent run
    pinned to the same browser tab.
    """

    if not config:
        return DEFAULT_SESSION_ID

    return (config.get("configurable") or {}).get("session_id", DEFAULT_SESSION_ID)


@tool
def request_human_input(prompt: str, input_type: str = "text") -> str:
    """Pause the task and ask the human for a value you cannot obtain
    yourself - e.g. an OTP / 2FA code, an email or SMS verification
    code, a CAPTCHA answer, a password, or a confirmation decision.

    Call this the moment the website asks for one of these instead of
    guessing or giving up. Execution stops until the human replies;
    their reply is returned to you as a string so you can type_text
    it into the right field and continue the task.

    Args:
        prompt: The exact question to show the human, with enough
            context to answer correctly (e.g. "Enter the 6-digit OTP
            sent to your phone to log into zepto.com").
        input_type: A hint about the kind of value expected, e.g.
            "otp", "password", "captcha", "confirmation", "text".
    """

    response = interrupt({
        "question": prompt,
        "input_type": input_type,
    })

    return response


@tool
@_log_tool_call
async def grant_media_permissions(url: str | None = None, config: RunnableConfig = None):
    """Allow camera and microphone access for the current browser session.

    Every session already grants camera/microphone access for all origins
    up front, so native browser permission prompts for them should never
    appear. Call this tool if a site still seems blocked from using the
    camera/microphone (e.g. before starting a video/voice call), optionally
    passing the site's URL to (re-)scope the grant to that origin. Never
    use request_human_input to ask the human whether to allow camera or
    microphone access - just grant it yourself.

    Args:
        url: The origin to scope the grant to, e.g. "https://meet.google.com".
            Omit to grant for all origins in this session.
    """

    try:
        await _grant_media_permissions(_session_id(config), url)

        return {
            "success": True,
            "permissions": ["camera", "microphone"],
            "url": url,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def open_page(url: str, config: RunnableConfig = None):
    """Open a webpage."""
    try:
        page = await get_page(_session_id(config))

        async def _goto():
            await page.goto(url)
            await page.wait_for_load_state("domcontentloaded")

        await _with_retries(_goto)

        return {
            "success": True,
            "url": page.url,
            "title": await page.title()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def observe_page(config: RunnableConfig = None):
    """List the visible, interactive elements on the current page.

    Each returned element includes a ready-to-use "selector" string.
    Always use that exact selector with click_element / type_text —
    do not write your own CSS selector, since several elements on a
    page often share the same tag, class or label.
    """

    page = await get_page(_session_id(config))
    elements = []

    locator = page.locator(
        "input, button, a, textarea, select, [role='button'], [onclick]"
    )

    count = await locator.count()

    for i in range(count):
        element = locator.nth(i)

        try:
            input_type = await element.get_attribute("type")
            tag = await element.evaluate("(el) => el.tagName.toLowerCase()")
            is_checkbox_like = tag == "input" and input_type in ("checkbox", "radio")

            visible = await element.is_visible()

            # Toggle/switch widgets (privacy, consent, marketing opt-in,
            # etc.) are almost always a real <input type="checkbox">
            # made visually invisible (opacity:0, sr-only, 1px, etc.) with
            # a styled <span>/<div> drawn on top as the switch the user
            # sees. A strict is_visible() check here used to drop that
            # real input entirely, leaving the agent with no selector it
            # could actually toggle - it would click the decorative span
            # instead, which often has no handler at all. So: never skip
            # a checkbox/radio input for invisibility, only for things
            # that are genuinely not on the page (display:none AND
            # zero-size, i.e. not just hidden-but-present).
            if not visible and is_checkbox_like:
                has_box = await element.evaluate(
                    """(el) => {
                        const r = el.getBoundingClientRect();
                        return r.width > 0 || r.height > 0 || el.offsetParent !== null;
                    }"""
                )
                if not has_box:
                    continue
            elif not visible:
                continue

            # Tag the exact DOM node so the selector we hand back can
            # never collide with another element that shares the same
            # tag/class/label (the cause of earlier "strict mode" click
            # failures on pages with duplicate/hidden lookalike elements).
            tag_id = f"ba-{i}"
            await element.evaluate(
                "(el, id) => el.setAttribute('data-ba-id', id)", tag_id
            )

            text = (await element.inner_text()).strip()

            checked = None
            if tag == "input" and input_type in ("checkbox", "radio"):
                checked = await element.is_checked()
            else:
                aria_checked = await element.get_attribute("aria-checked")
                if aria_checked is not None:
                    checked = aria_checked == "true"

            # Surface the label text a consent/T&C checkbox is paired with -
            # these are almost never self-describing (just an empty <input>),
            # the meaning lives in a <label>/sibling text node next to it,
            # which is what the agent needs to tell a required consent box
            # apart from an unrelated "subscribe to newsletter" checkbox.
            associated_label = None
            if tag == "input" and input_type in ("checkbox", "radio"):
                try:
                    associated_label = await element.evaluate(
                        """(el) => {
                            const byFor = el.id && document.querySelector(`label[for='${el.id}']`);
                            const wrapping = el.closest('label');
                            const label = byFor || wrapping;
                            if (label) return label.innerText.trim().slice(0, 200);
                            const parent = el.parentElement;
                            return parent ? parent.innerText.trim().slice(0, 200) : null;
                        }"""
                    )
                except Exception:
                    associated_label = None

            is_required = await element.get_attribute("required") is not None
            if not is_required:
                is_required = await element.get_attribute("aria-required") == "true"

            elements.append({
                "selector": f"[data-ba-id='{tag_id}']",
                "tag": tag,
                "type": input_type,
                "name": await element.get_attribute("name"),
                "text": text[:100] if text else None,
                "placeholder": await element.get_attribute("placeholder"),
                "aria_label": await element.get_attribute("aria-label"),
                "checked": checked,
                "required": is_required,
                "label": associated_label,
            })

        except Exception:
            # Element detached/changed mid-scan (e.g. page re-rendered) - skip it.
            continue

    return {
        "url": page.url,
        "title": await page.title(),
        "elements": elements
    }


@tool
@_log_tool_call
async def click_element(selector: str, config: RunnableConfig = None):
    """Click an element. Use the exact "selector" value returned by observe_page."""
    try:
        page = await get_page(_session_id(config))
        locator = page.locator(selector).first

        async def _click():
            await locator.wait_for(state="visible", timeout=10000)
            await locator.click()

        await _with_retries(_click)

        return {
            "success": True,
            "selector": selector,
            "url": page.url
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def type_text(selector: str, text: str, config: RunnableConfig = None):
    """Type text into an input. Use the exact "selector" value returned by observe_page."""

    try:
        page = await get_page(_session_id(config))
        locator = page.locator(selector).first

        async def _fill():
            await locator.wait_for(state="visible", timeout=10000)
            await locator.fill(text)

        await _with_retries(_fill)

        return {
            "success": True,
            "selector": selector,
            "message": "Text entered successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def press_key(key: str, selector: str | None = None, config: RunnableConfig = None):
    """Press a keyboard key, e.g. "Enter", "Tab", "Escape".

    If selector is given (use the exact value from observe_page),
    the element is focused first. Use this to submit a search box or
    form that has no visible/clickable submit button.
    """

    try:
        page = await get_page(_session_id(config))

        async def _press():
            if selector:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible", timeout=10000)
                await locator.press(key)
            else:
                await page.keyboard.press(key)

        await _with_retries(_press)

        return {
            "success": True,
            "key": key
        }

    except Exception as e:
        return {
            "success": False,
            "key": key,
            "error": str(e)
        }


@tool
@_log_tool_call
async def select_option(selector: str, value: str, config: RunnableConfig = None):
    """Choose an option in a <select> dropdown, by its visible label or value.
    Use the exact "selector" value returned by observe_page."""

    try:
        page = await get_page(_session_id(config))
        locator = page.locator(selector).first

        async def _select():
            await locator.wait_for(state="visible", timeout=10000)

            try:
                await locator.select_option(label=value)
            except Exception:
                await locator.select_option(value=value)

        await _with_retries(_select)

        return {
            "success": True,
            "selector": selector,
            "value": value
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def set_checkbox(selector: str, checked: bool = True, config: RunnableConfig = None):
    """Set a checkbox/toggle (e.g. a consent, Terms & Conditions, or
    privacy-policy checkbox) to a specific checked state.

    Always prefer this over click_element for checkboxes - it first reads
    the element's current state and only clicks if that state does not
    already match `checked`, so it is safe to call even if you are not
    sure whether the box is already ticked (clicking an already-checked
    box with click_element would wrongly uncheck it).

    Works for native <input type="checkbox"> as well as custom
    role="checkbox"/role="switch" toggle widgets.

    Args:
        selector: The exact "selector" value returned by observe_page.
        checked: True to check/tick the box, False to uncheck it.
    """

    try:
        page = await get_page(_session_id(config))
        locator = page.locator(selector).first

        # Use "attached" rather than "visible": toggle/consent checkboxes
        # are routinely implemented as a real <input> made permanently,
        # intentionally invisible (opacity:0 / sr-only / 1px) with a
        # styled span drawn on top as the switch. Waiting for "visible"
        # here would legitimately time out forever on that pattern even
        # though the element is perfectly interactive.
        await locator.wait_for(state="attached", timeout=10000)

        tag = await locator.evaluate("(el) => el.tagName.toLowerCase()")
        input_type = await locator.get_attribute("type")

        if tag == "input" and input_type in ("checkbox", "radio"):
            try:
                # force=True skips Playwright's visibility/actionability
                # checks - required for the hidden-native-input pattern
                # above, which would otherwise fail actionability.
                await locator.set_checked(checked, force=True)
                current = await locator.is_checked()
            except Exception:
                current = None

            if current != checked:
                # Fallback for elements with zero bounding box (true
                # display:none), where even a forced click has nothing to
                # click at: set the DOM property and dispatch the events
                # a real click would fire, so framework (React/Vue/etc.)
                # onChange handlers still see the change.
                await locator.evaluate(
                    """(el, val) => {
                        el.checked = val;
                        el.dispatchEvent(new Event('click', { bubbles: true }));
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    checked
                )
                current = await locator.is_checked()
        else:
            # Custom widget (e.g. role="switch" div with no backing
            # <input>) - toggle via click based on aria-checked, since
            # Playwright's set_checked only understands native inputs.
            aria_checked = await locator.get_attribute("aria-checked")
            current = aria_checked == "true"

            if current != checked:
                try:
                    await locator.click(force=True, timeout=5000)
                except Exception:
                    # Last resort: fire a synthetic click directly, which
                    # works even on elements Playwright can't hit-test
                    # (zero-size, covered, off-screen).
                    await locator.evaluate(
                        "(el) => el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))"
                    )

                await page.wait_for_timeout(150)
                aria_checked = await locator.get_attribute("aria-checked")
                current = aria_checked == "true" if aria_checked is not None else checked

        return {
            "success": current == checked,
            "selector": selector,
            "checked": current
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def fill_date(selector: str, date: str, config: RunnableConfig = None):
    """Set a date field, including JS date-picker widgets that reject
    plain typing/fill.

    Args:
        selector: The exact "selector" value returned by observe_page,
            for the date input/field (not the calendar popup itself).
        date: The target date in "YYYY-MM-DD" format, e.g. "2026-03-05".

    Handles three cases automatically:
      1. Native <input type="date"> - filled directly (Playwright
         requires ISO format for these, which is what `date` already is).
      2. A text input that simply accepts typed text (many date-range
         pickers do) - typed and confirmed with Enter/Escape.
      3. A text input that opens a calendar popup requiring clicks (e.g.
         react-datepicker, flatpickr, MUI, Ant Design, jQuery UI) - the
         popup is opened, then its month/year navigation is clicked
         forward/back until the target month is showing, then the
         matching day cell is clicked.
    """

    try:
        import re
        from datetime import datetime

        page = await get_page(_session_id(config))
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=10000)

        try:
            target = datetime.strptime(date.strip(), "%Y-%m-%d")
        except ValueError:
            return {
                "success": False,
                "selector": selector,
                "error": f"date must be in YYYY-MM-DD format, got {date!r}"
            }

        input_type = await locator.get_attribute("type")

        # Case 1: native date input.
        if input_type == "date":
            await locator.fill(target.strftime("%Y-%m-%d"))
            value = await locator.input_value()

            return {
                "success": value == target.strftime("%Y-%m-%d"),
                "selector": selector,
                "method": "native_date_input",
                "value": value
            }

        # Open whatever popup this field might trigger.
        await locator.click()
        await page.wait_for_timeout(300)

        popup_selectors = [
            "[role='dialog']",
            "[role='grid']",
            ".react-datepicker",
            ".flatpickr-calendar",
            ".daterangepicker",
            ".ui-datepicker",
            ".MuiPickersPopper-root",
            ".ant-picker-dropdown",
            "[class*='datepicker' i]",
            "[class*='date-picker' i]",
            "[class*='calendar' i]",
        ]

        popup = None
        for sel in popup_selectors:
            candidate = page.locator(sel).first
            try:
                if await candidate.is_visible(timeout=500):
                    popup = candidate
                    break
            except Exception:
                continue

        if popup is None:
            # Case 2: no popup appeared - this is a plain text field.
            await locator.fill(target.strftime("%Y-%m-%d"))
            await page.keyboard.press("Enter")

            return {
                "success": True,
                "selector": selector,
                "method": "typed_text",
                "value": await locator.input_value()
            }

        # Case 3: calendar popup. Click "next"/"prev" month controls until
        # the visible month/year header matches the target, then click the
        # target day. Month names are matched loosely (e.g. "March 2026",
        # "Mar 2026") since every library formats its header differently.
        month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        target_month_name = month_names[target.month - 1]
        target_year = str(target.year)

        header_selectors = [
            "[class*='header' i]", "[class*='month' i]", "[class*='caption' i]",
            "th[colspan]", "[role='presentation']",
        ]
        next_selectors = [
            "[aria-label*='next' i]", "button[class*='next' i]", "[class*='next' i]",
        ]
        prev_selectors = [
            "[aria-label*='prev' i]", "button[class*='prev' i]", "[class*='prev' i]",
        ]

        async def current_header_text() -> str:
            for sel in header_selectors:
                el = popup.locator(sel).first
                try:
                    if await el.is_visible(timeout=300):
                        txt = (await el.inner_text()).strip().lower()
                        if txt and any(m in txt for m in month_names):
                            return txt
                except Exception:
                    continue
            try:
                return (await popup.inner_text()).strip().lower()
            except Exception:
                return ""

        matched = False
        for _ in range(36):  # up to 3 years of month-stepping, either direction
            header = await current_header_text()

            if target_month_name in header and target_year in header:
                matched = True
                break

            direction_selectors = next_selectors
            if header:
                # If we can parse a year from the header and it's beyond
                # target, step backward instead of forward.
                year_match = re.search(r"(19|20)\d{2}", header)
                if year_match and int(year_match.group(0)) > target.year:
                    direction_selectors = prev_selectors

            clicked = False
            for sel in direction_selectors:
                btn = popup.locator(sel).first
                try:
                    if await btn.is_visible(timeout=300):
                        await btn.click()
                        await page.wait_for_timeout(200)
                        clicked = True
                        break
                except Exception:
                    continue

            if not clicked:
                break

        if not matched:
            # Never guess a day in whatever month happens to be showing -
            # that would silently select the wrong date instead of
            # reporting a clear failure the agent can act on (e.g. the
            # target date is outside the picker's allowed range).
            return {
                "success": False,
                "selector": selector,
                "method": "calendar_popup",
                "month_matched": False,
                "day_clicked": False,
                "error": (
                    f"Could not navigate the calendar to {target_month_name} "
                    f"{target_year} - it may be outside the range this "
                    "picker allows (e.g. a past date, or too far in the "
                    "future)."
                ),
            }

        day_str = str(target.day)
        day_candidates = popup.locator(
            f"[role='gridcell']:has-text('{day_str}'), "
            f"td:has-text('{day_str}'), "
            f"button:has-text('{day_str}')"
        )

        count = await day_candidates.count()
        clicked_day = False

        for i in range(count):
            cell = day_candidates.nth(i)
            try:
                cell_text = (await cell.inner_text()).strip()
                class_attr = (await cell.get_attribute("class")) or ""

                if cell_text != day_str:
                    continue
                if any(bad in class_attr.lower() for bad in [
                    "outside", "disabled", "other-month", "not-current"
                ]):
                    continue
                if not await cell.is_visible(timeout=300):
                    continue

                await cell.click()
                clicked_day = True
                break
            except Exception:
                continue

        await page.wait_for_timeout(200)

        return {
            "success": clicked_day,
            "selector": selector,
            "method": "calendar_popup",
            "month_matched": True,
            "day_clicked": clicked_day,
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def check_form_requirements(config: RunnableConfig = None):
    """Scan the current page for a form's required consent/Terms &
    Conditions checkboxes and required fields that are not yet satisfied.

    Call this right before submitting any form (signup, checkout, contact,
    application, etc.) to avoid a submission that silently fails or gets
    rejected by client-side validation because an "I agree to the Terms
    and Conditions" / "I consent to..." / "I accept the Privacy Policy"
    checkbox was left unticked, or a required field is empty.

    Returns a list of problems to fix (each with a selector you can act
    on via set_checkbox or type_text) - empty list means the form looks
    ready to submit.
    """

    try:
        page = await get_page(_session_id(config))
        problems = []

        consent_keywords = [
            "terms", "conditions", "privacy", "consent", "agree",
            "accept", "policy", "gdpr", "acknowledge",
        ]

        checkbox_locator = page.locator(
            "input[type='checkbox'], [role='checkbox'], [role='switch']"
        )
        count = await checkbox_locator.count()

        for i in range(count):
            el = checkbox_locator.nth(i)

            try:
                # Same hidden-native-input toggle pattern as observe_page -
                # don't drop a checkbox/radio just because it's visually
                # hidden behind a styled switch; only drop it if it truly
                # has no box at all (not rendered anywhere on the page).
                input_type_probe = await el.get_attribute("type")
                is_checkbox_like = input_type_probe in ("checkbox", "radio")

                if not await el.is_visible(timeout=300):
                    if not is_checkbox_like:
                        continue

                    has_box = await el.evaluate(
                        """(el) => {
                            const r = el.getBoundingClientRect();
                            return r.width > 0 || r.height > 0 || el.offsetParent !== null;
                        }"""
                    )
                    if not has_box:
                        continue

                label_text = await el.evaluate(
                    """(el) => {
                        const byFor = el.id && document.querySelector(`label[for='${el.id}']`);
                        const wrapping = el.closest('label');
                        const label = byFor || wrapping;
                        if (label) return label.innerText.trim();
                        const parent = el.parentElement;
                        return parent ? parent.innerText.trim() : '';
                    }"""
                )
                label_text = (label_text or "").strip()

                is_required = await el.get_attribute("required") is not None
                is_required = is_required or await el.get_attribute("aria-required") == "true"
                looks_like_consent = any(
                    kw in label_text.lower() for kw in consent_keywords
                )

                if not (is_required or looks_like_consent):
                    continue

                input_type = await el.get_attribute("type")
                if input_type in ("checkbox", "radio"):
                    is_checked = await el.is_checked()
                else:
                    is_checked = (await el.get_attribute("aria-checked")) == "true"

                if not is_checked:
                    tag_id = f"ba-req-{i}"
                    await el.evaluate(
                        "(el, id) => el.setAttribute('data-ba-id', id)", tag_id
                    )

                    problems.append({
                        "type": "unchecked_consent_or_required_checkbox",
                        "selector": f"[data-ba-id='{tag_id}']",
                        "label": label_text[:150],
                    })
            except Exception:
                continue

        required_inputs = page.locator(
            "input[required], textarea[required], select[required], "
            "[aria-required='true']"
        )
        count = await required_inputs.count()

        for i in range(count):
            el = required_inputs.nth(i)

            try:
                if not await el.is_visible(timeout=300):
                    continue

                tag = await el.evaluate("(el) => el.tagName.toLowerCase()")
                input_type = await el.get_attribute("type")

                if tag == "input" and input_type in ("checkbox", "radio"):
                    continue  # already covered above

                value = await el.input_value() if tag in ("input", "textarea", "select") else ""

                if not value.strip():
                    tag_id = f"ba-req-field-{i}"
                    await el.evaluate(
                        "(el, id) => el.setAttribute('data-ba-id', id)", tag_id
                    )

                    problems.append({
                        "type": "empty_required_field",
                        "selector": f"[data-ba-id='{tag_id}']",
                        "name": await el.get_attribute("name"),
                        "placeholder": await el.get_attribute("placeholder"),
                    })
            except Exception:
                continue

        return {
            "success": True,
            "ready_to_submit": len(problems) == 0,
            "problems": problems,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def submit_form(selector: str, config: RunnableConfig = None):
    """Click a form's submit button and verify the submission actually
    went through, instead of assuming a click = success.

    After clicking, this waits briefly and then checks whether the page
    still shows validation/error messages (e.g. "required", "please
    agree", text in [role='alert'] or .error elements) or whether the
    URL/content changed as expected. Use this instead of plain
    click_element for the final submit action of any form.

    If it reports unresolved problems, call check_form_requirements to
    see what's missing (e.g. an unchecked consent box), fix them, and
    call submit_form again - do not declare the task done after a single
    failed attempt.

    Args:
        selector: The exact "selector" value (from observe_page) for the
            submit button.
    """

    try:
        page = await get_page(_session_id(config))
        locator = page.locator(selector).first

        async def _click():
            await locator.wait_for(state="visible", timeout=10000)
            await locator.click()

        url_before = page.url

        await _with_retries(_click)
        await page.wait_for_timeout(1000)

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass

        url_after = page.url
        navigated = url_after != url_before

        error_texts = []
        error_locator = page.locator(
            "[role='alert'], .error, [class*='error' i], [class*='invalid' i], "
            "[aria-invalid='true']"
        )
        count = await error_locator.count()

        for i in range(min(count, 20)):
            el = error_locator.nth(i)
            try:
                if await el.is_visible(timeout=200):
                    txt = (await el.inner_text()).strip()
                    if txt:
                        error_texts.append(txt[:200])
            except Exception:
                continue

        likely_succeeded = navigated or not error_texts

        return {
            "success": likely_succeeded,
            "selector": selector,
            "navigated": navigated,
            "url_before": url_before,
            "url_after": url_after,
            "visible_errors": error_texts,
            "message": (
                "Submission appears to have gone through."
                if likely_succeeded else
                "Form still shows validation errors after submit - fix "
                "them (check check_form_requirements) and try again."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "selector": selector,
            "error": str(e)
        }


@tool
@_log_tool_call
async def scroll_page(direction: str = "down", amount: int = 800, config: RunnableConfig = None):
    """Scroll the page. direction: "up", "down", "top", or "bottom".

    Use this before observe_page/take_screenshot to reveal elements
    below the fold or to trigger lazy-loaded content on long pages.
    """

    try:
        page = await get_page(_session_id(config))

        if direction == "top":
            await page.evaluate("window.scrollTo(0, 0)")
        elif direction == "bottom":
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        elif direction == "up":
            await page.evaluate(f"window.scrollBy(0, -{amount})")
        else:
            await page.evaluate(f"window.scrollBy(0, {amount})")

        await page.wait_for_timeout(300)

        return {
            "success": True,
            "direction": direction
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def go_back(config: RunnableConfig = None):
    """Navigate to the previous page in browser history."""

    try:
        page = await get_page(_session_id(config))
        await page.go_back(wait_until="domcontentloaded")

        return {
            "success": True,
            "url": page.url,
            "title": await page.title()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def go_forward(config: RunnableConfig = None):
    """Navigate forward in browser history."""

    try:
        page = await get_page(_session_id(config))
        await page.go_forward(wait_until="domcontentloaded")

        return {
            "success": True,
            "url": page.url,
            "title": await page.title()
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def close_popup(config: RunnableConfig = None):
    """Detect and close a popup, modal dialog, cookie-consent banner,
    or overlay that is currently blocking the page.

    IMPORTANT: Only call this tool if the human's instructions
    explicitly asked you to close, dismiss, or accept a popup/banner.
    If the human did not mention popups at all, do NOT call this
    tool - leave the page exactly as it is, even if a popup is
    visible, and proceed with the rest of the task around it.
    """

    try:
        page = await get_page(_session_id(config))

        # Real-world popups rarely expose a consistent container (many are
        # custom React/styled-components modals with hashed class names and
        # no role="dialog"). So search page-wide for a close control first,
        # by accessible role/name/aria-label/glyph - this catches far more
        # sites than matching a container class name ever will.
        close_control_selectors = [
            "[aria-label='Close' i]",
            "[aria-label='Dismiss' i]",
            "[aria-label*='close' i]",
            "button[class*='close' i]",
            "span[class*='close' i]",
            "a[class*='close' i]",
            "[data-dismiss]",
            "[data-testid*='close' i]",
            "button:has-text('No, thanks')",
            "button:has-text('No thanks')",
            "button:has-text('Got it')",
            "button:has-text('Dismiss')",
            "button:has-text('Close')",
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "#onetrust-accept-btn-handler",
        ]

        for sel in close_control_selectors:
            btn = page.locator(sel).first

            try:
                if await btn.is_visible(timeout=800):
                    await btn.click()
                    await page.wait_for_timeout(300)

                    return {
                        "success": True,
                        "method": "clicked_close_button",
                        "selector": sel
                    }
            except Exception:
                continue

        # Common custom close icons rendered as bare glyph text (×, X, ✕)
        # with no aria-label - match the exact visible text of the control.
        for glyph in ["×", "✕", "✖"]:
            btn = page.get_by_text(glyph, exact=True).first

            try:
                if await btn.is_visible(timeout=800):
                    await btn.click()
                    await page.wait_for_timeout(300)

                    return {
                        "success": True,
                        "method": "clicked_close_glyph",
                        "glyph": glyph
                    }
            except Exception:
                continue

        # No identifiable close control - if a dialog/overlay is at least
        # detectable, try Escape as a last resort.
        dialog_selectors = [
            "[role='dialog']",
            "[role='alertdialog']",
            "[aria-modal='true']",
            "#onetrust-banner-sdk",
        ]

        for sel in dialog_selectors:
            container = page.locator(sel).first

            try:
                if not await container.is_visible(timeout=800):
                    continue
            except Exception:
                continue

            try:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(300)

                if not await container.is_visible(timeout=800):
                    return {
                        "success": True,
                        "method": "escape_key",
                        "container": sel
                    }
            except Exception:
                pass

        return {
            "success": False,
            "message": "No popup, dialog, or overlay was found on the page"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def extract_text(selector: str | None = None, config: RunnableConfig = None):
    """Read the visible text of the page, or of one element if a
    selector is given (use the exact value from observe_page).

    Cheaper and more reliable than take_screenshot when you only
    need to read content - e.g. prices, confirmation messages,
    search results, or error banners.
    """

    try:
        page = await get_page(_session_id(config))

        if selector:
            locator = page.locator(selector).first
            await locator.wait_for(state="visible", timeout=10000)
            text = await locator.inner_text()
        else:
            text = await page.locator("body").inner_text()

        text = text.strip()

        return {
            "success": True,
            "text": text[:5000],
            "truncated": len(text) > 5000
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def wait(seconds: float = 1.0, config: RunnableConfig = None):
    """Pause briefly (max 10s) to let dynamic content finish loading
    before the next action, e.g. after a search or an AJAX update."""

    try:
        page = await get_page(_session_id(config))
        seconds = max(0.0, min(seconds, 10.0))
        await page.wait_for_timeout(seconds * 1000)

        return {
            "success": True,
            "waited_seconds": seconds
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def solve_captcha(auto_fill: bool = True, config: RunnableConfig = None):
    """Detect and try to solve a CAPTCHA on the current page, for free,
    using local OCR - no paid service, no API key, no network call.

    Only works for plain text/number image CAPTCHAs (the common kind
    used on login/signup/contact forms). reCAPTCHA and hCaptcha are
    built specifically to resist automated solving, so this tool cannot
    solve those - when it detects one of those instead, call
    request_human_input to get a human past it rather than retrying
    this tool.

    Args:
        auto_fill: If True (default) and an image captcha is solved,
            also type the answer into the nearest text input next to
            the captcha image. Set False to just return the answer
            and fill it yourself with type_text.
    """

    try:
        page = await get_page(_session_id(config))
        detected = await _captcha.detect_captcha(page)

        if detected is None:
            return {
                "success": True,
                "captcha_found": False,
                "message": "No captcha was detected on the current page.",
            }

        result = await _captcha.solve(page, detected)

        if not result["solved"]:
            return {
                "success": False,
                "captcha_found": True,
                "captcha_type": detected["type"],
                "solved": False,
                "reason": result["reason"],
            }

        answer = result["answer"]
        filled_selector = None

        if auto_fill:
            image_locator = page.locator(detected["selector"]).first
            input_locator = image_locator.locator(
                "xpath=following::input[1]"
            ).first

            try:
                await input_locator.wait_for(state="visible", timeout=3000)
                await input_locator.fill(answer)
                filled_selector = detected["selector"] + " ~ input"
            except Exception:
                filled_selector = None

        return {
            "success": True,
            "captcha_found": True,
            "captcha_type": detected["type"],
            "solved": True,
            "answer": answer,
            "auto_filled": filled_selector is not None,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def list_frames_and_canvases(config: RunnableConfig = None):
    """List <iframe> and <canvas> elements on the current page, each
    with its on-screen bounding box.

    Many embedded games/widgets (e.g. CrazyGames-style game embeds)
    render inside an <iframe> containing a <canvas>, which exposes no
    clickable DOM elements of its own - observe_page will never find
    anything usable inside them. Call this tool to find the iframe/canvas
    region on screen, then use click_at_coordinates with a point inside
    the returned bounding box (e.g. its "center") to interact with it.
    """

    try:
        page = await get_page(_session_id(config))
        results = []

        locator = page.locator("iframe, canvas")
        count = await locator.count()

        for i in range(count):
            element = locator.nth(i)

            try:
                if not await element.is_visible():
                    continue

                box = await element.bounding_box()
                if box is None:
                    continue

                tag = await element.evaluate("(el) => el.tagName.toLowerCase()")
                tag_id = f"ba-frame-{i}"
                await element.evaluate(
                    "(el, id) => el.setAttribute('data-ba-id', id)", tag_id
                )

                results.append({
                    "selector": f"[data-ba-id='{tag_id}']",
                    "tag": tag,
                    "src": await element.get_attribute("src") if tag == "iframe" else None,
                    "bounding_box": box,
                    "center": {
                        "x": box["x"] + box["width"] / 2,
                        "y": box["y"] + box["height"] / 2,
                    },
                })
            except Exception:
                continue

        return {
            "success": True,
            "url": page.url,
            "frames_and_canvases": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def click_at_coordinates(
    x: float, y: float, click_count: int = 1, config: RunnableConfig = None
):
    """Click at a specific pixel coordinate in the page viewport,
    bypassing the DOM entirely.

    Use this for canvas-rendered content (games, drawing apps, maps)
    and anything inside an iframe that observe_page/click_element
    cannot target because there is no clickable DOM element at that
    spot - only pixels. Call list_frames_and_canvases first to find the
    bounding box of the canvas/iframe, then pick a point inside it
    (e.g. its "center") to pass here.

    Args:
        x: Horizontal pixel offset from the left edge of the viewport.
        y: Vertical pixel offset from the top edge of the viewport.
        click_count: Number of clicks to fire at this point - use 2 for
            a double-click. Defaults to 1.
    """

    try:
        page = await get_page(_session_id(config))
        await page.mouse.click(x, y, click_count=click_count)

        return {
            "success": True,
            "x": x,
            "y": y,
            "click_count": click_count,
        }

    except Exception as e:
        return {
            "success": False,
            "x": x,
            "y": y,
            "error": str(e)
        }


def _safe_filename(name: str) -> str:
    """Strip any directory components from a server-/URL-suggested
    filename before it's used to build a path - both download.suggested_filename
    and a URL's basename are untrusted (attacker-controlled) strings, and
    without this a name like "../../etc/passwd" would let a download
    escape the downloads/ directory."""

    return Path(name).name or "downloaded_file"


@tool
@_log_tool_call
async def download_file(
    url: str | None = None,
    selector: str | None = None,
    filename: str | None = None,
    config: RunnableConfig = None,
):
    """Download a file (image, PDF, DOCX, ZIP, etc.) from the current
    browser session to the local downloads/ directory.

    Args:
        url: Direct URL of the file to download (e.g. an image's "src",
            a PDF link, a "raw file" link). Use this when you already
            know the file's URL - e.g. an "href"/"src" seen via
            observe_page or extract_text - and don't need to click
            anything first.
        selector: The exact "selector" value from observe_page for a
            download link/button to click (e.g. a "Download" button
            that triggers a browser download rather than linking
            straight to a file). Use this instead of `url` when the
            file has no plain URL you can fetch directly.
        filename: Optional name to save the file as. If omitted, the
            server-suggested filename (or the URL's basename) is used.

    Exactly one of `url` or `selector` must be given.
    """

    if not url and not selector:
        return {"success": False, "error": "Provide either url or selector."}

    if url and selector:
        return {
            "success": False,
            "error": "Provide only one of url or selector, not both."
        }

    try:
        page = await get_page(_session_id(config))
        downloads_dir = Path("downloads")
        downloads_dir.mkdir(exist_ok=True)

        download = None

        if selector:
            locator = page.locator(selector).first
            await locator.wait_for(state="visible", timeout=10000)

            async with page.expect_download(timeout=15000) as download_info:
                await locator.click()

            download = await download_info.value
        else:
            # Many direct file links (images, PDFs) don't fire a browser
            # "download" event at all - the browser just navigates to /
            # renders them inline. Try triggering a real download first
            # (covers links the server marks Content-Disposition:
            # attachment), then fall back to fetching the bytes directly.
            try:
                async with page.expect_download(timeout=8000) as download_info:
                    await page.evaluate(
                        """(u) => {
                            const a = document.createElement('a');
                            a.href = u;
                            a.download = '';
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                        }""",
                        url,
                    )

                download = await download_info.value
            except Exception:
                download = None

        if download is not None:
            save_name = _safe_filename(filename or download.suggested_filename)
            filepath = downloads_dir / save_name
            await download.save_as(str(filepath))

            return {
                "success": True,
                "path": str(filepath),
                "url": download.url,
                "method": "browser_download",
            }

        # Fallback: fetch the bytes directly through the session's own
        # request context (reuses cookies/auth already in the browser
        # session for that origin).
        response = await page.context.request.get(url)

        if not response.ok:
            return {
                "success": False,
                "error": f"HTTP {response.status} fetching {url}",
            }

        body = await response.body()

        if not filename:
            from urllib.parse import unquote, urlparse
            filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])

        save_name = _safe_filename(filename)
        filepath = downloads_dir / save_name
        filepath.write_bytes(body)

        return {
            "success": True,
            "path": str(filepath),
            "url": url,
            "method": "direct_fetch",
            "content_type": response.headers.get("content-type"),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
@_log_tool_call
async def take_screenshot(filename: str | None = None, config: RunnableConfig = None):
    """Take a screenshot of the current page.

    Args:
        filename: Optional name for the screenshot file. If omitted, a
            name unique to this browser session is generated automatically
            - safe to leave unset even when multiple sub-agents are taking
            screenshots in parallel, since they will never collide.
    """

    try:
        session_id = _session_id(config)
        page = await get_page(session_id)
        await page.wait_for_load_state("domcontentloaded")
        await page.wait_for_selector("body")

        screenshot_dir = Path("screenshots")
        screenshot_dir.mkdir(exist_ok=True)

        if filename is None:
            filename = f"page_{session_id[:8]}.png"

        if not Path(filename).suffix:
            filename += ".png"

        filepath = screenshot_dir / filename

        await page.screenshot(
            path=str(filepath),
            full_page=True
        )

        return {
            "success": True,
            "path": str(filepath),
            "url": page.url
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

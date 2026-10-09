"""Free, local CAPTCHA solving.

Only plain text/number image CAPTCHAs (the kind many sites use for
simple bot-deterrence on forms, logins, etc.) can be solved this way,
using local OCR (Tesseract, via pytesseract - no account, no API key,
no network call). This is the free equivalent of what a paid solving
service would otherwise charge per solve for the same case.

reCAPTCHA and hCaptcha are purpose-built to resist exactly this kind of
automated solving (that's their whole job), and there is no legitimate
free way to defeat them - any such trick is an anti-abuse bypass, not a
CAPTCHA "solver". detect_captcha still recognizes them so the agent can
immediately fall back to request_human_input instead of getting stuck.

Cloudflare Turnstile's "verify you are human" widget is different: in
its default (non-interactive) mode it is just a checkbox a real visitor
clicks to confirm a browser is present, and clicking it is the designed
interaction, not a bypass of anything (the same category as ticking a
cookie-consent box). This module clicks that checkbox and waits for the
widget to report success. If Cloudflare escalates to a harder
interactive puzzle, that part still can't be solved locally and falls
back the same way reCAPTCHA/hCaptcha do.
"""

import asyncio
import io

from playwright.async_api import Page

try:
    import pytesseract
    from PIL import Image, ImageOps

    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class CaptchaSolverError(Exception):
    pass


def _preprocess(image_bytes: bytes) -> "Image.Image":
    """Clean up a typical distorted-text CAPTCHA image to make OCR viable:
    greyscale, autocontrast, and a binary threshold to strip background
    noise/lines while keeping glyph edges crisp."""

    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.point(lambda p: 255 if p > 140 else 0)

    scale = 3
    image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)

    return image


def _ocr_sync(image_bytes: bytes) -> str:
    if not _OCR_AVAILABLE:
        raise CaptchaSolverError(
            "OCR solving requires the 'pytesseract' + 'Pillow' packages "
            "and the 'tesseract-ocr' system binary installed."
        )

    image = _preprocess(image_bytes)

    text = pytesseract.image_to_string(
        image,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    )

    return "".join(text.split())


async def detect_captcha(page: Page) -> dict | None:
    """Inspect the page for a known CAPTCHA widget and return its type
    plus the metadata needed to handle it, or None if none is found."""

    recaptcha = await page.evaluate(
        """() => {
            const el = document.querySelector('.g-recaptcha[data-sitekey]');
            if (el) return { sitekey: el.dataset.sitekey, size: el.dataset.size || null };
            const iframe = document.querySelector("iframe[src*='recaptcha']");
            if (iframe) {
                const match = iframe.src.match(/[?&]k=([^&]+)/);
                return match ? { sitekey: match[1], size: null } : null;
            }
            return null;
        }"""
    )

    if recaptcha and recaptcha.get("sitekey"):
        version = "v3" if recaptcha.get("size") == "invisible" else "v2"
        return {"type": f"recaptcha_{version}", "sitekey": recaptcha["sitekey"], "solvable_locally": False}

    hcaptcha = await page.evaluate(
        """() => {
            const el = document.querySelector('.h-captcha[data-sitekey]');
            if (el) return { sitekey: el.dataset.sitekey };
            const iframe = document.querySelector("iframe[src*='hcaptcha']");
            if (iframe) {
                const match = iframe.src.match(/[?&]sitekey=([^&]+)/);
                return match ? { sitekey: match[1] } : null;
            }
            return null;
        }"""
    )

    if hcaptcha and hcaptcha.get("sitekey"):
        return {"type": "hcaptcha", "sitekey": hcaptcha["sitekey"], "solvable_locally": False}

    turnstile = await page.evaluate(
        """() => {
            const el = document.querySelector(
                '.cf-turnstile[data-sitekey], div[data-sitekey][class*="turnstile" i]'
            );
            const iframe = document.querySelector(
                "iframe[src*='challenges.cloudflare.com']"
            );
            if (el || iframe) return { sitekey: el ? el.dataset.sitekey : null };
            return null;
        }"""
    )

    if turnstile is not None:
        return {
            "type": "turnstile",
            "sitekey": turnstile.get("sitekey"),
            "solvable_locally": True,
        }

    image_selector = (
        "img[src*='captcha' i], img[alt*='captcha' i], "
        "canvas[id*='captcha' i], img[id*='captcha' i]"
    )
    image_locator = page.locator(image_selector).first

    try:
        if await image_locator.is_visible(timeout=500):
            return {"type": "image", "selector": image_selector, "solvable_locally": True}
    except Exception:
        pass

    return None


async def _solve_turnstile(page: Page) -> dict:
    """Click Cloudflare Turnstile's "verify you are human" checkbox and
    wait for the widget to report success (a populated
    cf-turnstile-response token). The widget itself lives inside an
    iframe from challenges.cloudflare.com, and the checkbox inside that
    is sometimes rendered in a further nested frame, so a few selector
    strategies are tried."""

    widget_frame = page.frame_locator("iframe[src*='challenges.cloudflare.com']").first

    checkbox_candidates = [
        widget_frame.locator("input[type='checkbox']"),
        widget_frame.frame_locator("iframe").first.locator("input[type='checkbox']"),
        widget_frame.get_by_role("checkbox"),
        widget_frame.locator("label"),
    ]

    clicked = False
    for candidate in checkbox_candidates:
        try:
            await candidate.first.click(timeout=4000)
            clicked = True
            break
        except Exception:
            continue

    if not clicked:
        try:
            box = await page.locator(
                "iframe[src*='challenges.cloudflare.com']"
            ).first.bounding_box()
            if box:
                await page.mouse.click(
                    box["x"] + 30, box["y"] + box["height"] / 2
                )
                clicked = True
        except Exception:
            pass

    if not clicked:
        return {
            "solved": False,
            "reason": (
                "Could not locate the Turnstile checkbox to click. Ask the "
                "human for help instead (request_human_input)."
            ),
        }

    try:
        await page.wait_for_function(
            """() => {
                const el = document.querySelector('[name="cf-turnstile-response"]');
                return !!(el && el.value);
            }""",
            timeout=10000,
        )
        return {"solved": True, "answer": "checkbox_verified"}
    except Exception:
        return {
            "solved": False,
            "reason": (
                "Clicked the Turnstile checkbox but it did not pass "
                "automatically - Cloudflare likely escalated to an "
                "interactive puzzle challenge, which can't be solved "
                "locally. Ask the human for help instead "
                "(request_human_input)."
            ),
        }


async def solve(page: Page, captcha: dict) -> dict:
    """Attempt to solve a CAPTCHA previously found by detect_captcha.

    Returns {"solved": bool, "answer"?: str, "reason"?: str}. For
    reCAPTCHA/hCaptcha, "solved" is always False with a reason explaining
    why - the caller should fall back to request_human_input."""

    captcha_type = captcha["type"]

    if captcha_type == "turnstile":
        return await _solve_turnstile(page)

    if not captcha.get("solvable_locally"):
        return {
            "solved": False,
            "reason": (
                f"{captcha_type} cannot be solved by free local OCR - it is "
                "designed specifically to resist automated solving. Ask the "
                "human for help instead (request_human_input)."
            ),
        }

    locator = page.locator(captcha["selector"]).first
    screenshot_bytes = await locator.screenshot()

    answer = await asyncio.to_thread(_ocr_sync, screenshot_bytes)

    if not answer:
        return {
            "solved": False,
            "reason": "OCR could not read any text from the captcha image.",
        }

    return {"solved": True, "answer": answer}

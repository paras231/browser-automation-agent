import asyncio
import uuid
from datetime import datetime
from typing import List, TypedDict
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

try:
    from .tools import (
        open_page,
        observe_page,
        click_element,
        type_text,
        press_key,
        select_option,
        set_checkbox,
        fill_date,
        check_form_requirements,
        submit_form,
        scroll_page,
        go_back,
        go_forward,
        close_popup,
        extract_text,
        wait,
        take_screenshot,
        request_human_input,
        grant_media_permissions,
        solve_captcha,
        list_frames_and_canvases,
        click_at_coordinates,
        download_file,
        inspect_element,
        get_console_logs,
        get_network_requests,
        switch_devtools_tab,
        diagnose_page_issue,
    )
    from .browser import new_session, close_session, DEFAULT_SESSION_ID
    from .guardrails import check_request_safety
except ImportError:
    from tools import (
        open_page,
        observe_page,
        click_element,
        type_text,
        press_key,
        select_option,
        set_checkbox,
        fill_date,
        check_form_requirements,
        submit_form,
        scroll_page,
        go_back,
        go_forward,
        close_popup,
        extract_text,
        wait,
        take_screenshot,
        request_human_input,
        grant_media_permissions,
        solve_captcha,
        list_frames_and_canvases,
        click_at_coordinates,
        download_file,
        inspect_element,
        get_console_logs,
        get_network_requests,
        switch_devtools_tab,
        diagnose_page_issue,
    )
    from browser import new_session, close_session, DEFAULT_SESSION_ID
    from guardrails import check_request_safety

load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")


llm = ChatOllama(
    model="gemma4:31b",
    base_url="https://ollama.com",
     client_kwargs={
        "headers": {
            "Authorization": F"Bearer {OLLAMA_API_KEY}"
        }
    }
    # Configure authentication according to your
    # current Ollama Cloud setup.
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _worker_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")

    return f"""
Today's date is {today}. Resolve any relative or bare date in the
task (e.g. "9 Oct", "next Friday", "tomorrow") against this date -
never guess or assume a different year.
""" + WORKER_SYSTEM_PROMPT_BODY


WORKER_SYSTEM_PROMPT_BODY = """
You are a browser automation agent.

Use browser tools to accomplish the user's task.

After opening or changing a webpage, call observe_page
when you need to find something to click or type into.
Always use the exact "selector" value observe_page gives
you - never invent your own CSS selector.

If a search box or form has no visible submit button, use
press_key with key="Enter" (and the input's selector) to
submit it, rather than guessing at a button selector.

If the element you need is not in observe_page's results,
it may be below the fold or not yet loaded - use scroll_page
or wait before giving up.

If you need to interact with a canvas-rendered widget (e.g. a
game) or anything inside an iframe that observe_page finds no
usable elements for, call list_frames_and_canvases to get its
on-screen bounding box, then use click_at_coordinates with a
point inside it (e.g. its "center") instead of click_element.

To save a file (image, PDF, DOCX, ZIP, etc.) to disk, use
download_file - pass `url` if you already know the file's
direct link (e.g. an image "src" or a PDF "href"), or
`selector` (from observe_page) to click a download button/link
that triggers a browser download. Files are saved under
downloads/.

Prefer extract_text over take_screenshot when you only need
to read content (prices, confirmation text, results) - it is
cheaper and more reliable than reading an image.

FORM FILLING - be thorough, this is a common source of failure:
- For any checkbox (consent, Terms & Conditions, privacy policy,
  marketing opt-in, etc.), always use set_checkbox instead of
  click_element - it reads the current checked state first and
  only clicks if needed, so it never accidentally unchecks a box
  that was already correct.
- Before clicking the final submit/continue button of ANY form
  (signup, checkout, contact, application, booking, etc.), call
  check_form_requirements. It finds unchecked required/consent
  checkboxes and empty required fields that would otherwise cause
  a silent or rejected submission. Fix every problem it reports
  (set_checkbox / type_text) before submitting.
- Use submit_form (not click_element) for the actual submit click.
  It verifies the submission actually went through by checking for
  leftover validation errors on the page. If it reports the
  submission did not succeed, call check_form_requirements again,
  fix what it finds, and call submit_form again - do not stop after
  one failed attempt or claim success when submit_form reports
  failure.
- For date fields (date of birth, booking dates, deadlines, etc.),
  always use fill_date with the date in YYYY-MM-DD format rather
  than type_text or click_element - it handles native date inputs
  and also drives JS calendar/date-picker popups (clicking the
  right month/year and day) that reject plain typing.

Use go_back / go_forward for browser history navigation
instead of re-opening a URL when possible.

Only call close_popup if the human's instructions explicitly
ask you to close, dismiss, or accept a popup, banner, or
overlay. If they did not mention this, leave any popup alone
even if one is visible - do not call close_popup on your own
judgment.

Camera and microphone access is already allowed for every
site in your browser session - a native permission prompt for
them will never appear. If a site's own UI (not a browser
prompt) asks you to allow camera/microphone before a call or
voice message, call grant_media_permissions and proceed; never
ask the human for this.

Do not claim an action succeeded unless the tool result
confirms it.

If you encounter a CAPTCHA, call solve_captcha first. It can
solve plain text/number image captchas for free using local
OCR and will tell you if it filled the answer in for you. It
also handles Cloudflare Turnstile's "verify you are human"
checkbox by clicking it directly. It cannot solve reCAPTCHA or
hCaptcha (nothing free legitimately can), nor a Turnstile
challenge that escalates past the checkbox into an interactive
puzzle - when it reports one of those, or if it fails to solve
an image captcha, immediately call request_human_input instead
of retrying it yourself.

If the website asks for something only the human can
provide - an OTP / 2FA code, an email or SMS verification
code, a CAPTCHA answer it could not solve, a password, or a
yes/no confirmation for a sensitive action - call
request_human_input with a clear question instead of guessing,
making one up, or giving up. Wait for the returned value, then
use it (e.g. type_text it into the right field) and continue
the task.

DEBUGGING A BROKEN PAGE - use this when something doesn't work as
expected (a blank section, a button that does nothing, missing or
stale content, a layout glitch, or any mismatch between what the
task expects and what the page actually does):
- Call diagnose_page_issue (pass the element's selector if you've
  already spotted which one looks broken). It gathers, in one call,
  a screenshot of the current page, recent JS console
  errors/warnings and uncaught exceptions, failed/erroring network
  requests, and - if a selector was given - that element's HTML,
  computed styles, and on-screen box. It does not decide what's
  wrong for you; it only collects the evidence.
- Review everything it returned together to work out the actual
  root cause: a console error naming a null/undefined property or a
  failed fetch usually explains a feature that silently does
  nothing; a failed_request with a 4xx/5xx status or a network
  failure reason explains missing/stale data; an element whose
  computed_style shows display:none/visibility:hidden/opacity:0/a
  zero-size bounding box when it should be visible, or a position
  overlapped by something else, explains a "missing" or
  mis-rendered element.
- Then explain the root cause and give clear, step-by-step
  instructions for how to fix it (e.g. "the signup button's click
  handler throws TypeError: Cannot read properties of null (reading
  'value') at line X because it reads #email before the field
  exists in the DOM - move the handler so it runs after the form
  renders, or null-check the element first"). You cannot edit the
  site's code yourself, so your job here is an accurate diagnosis
  and fix instructions, not applying the fix.
- You can also open one devtools-style panel at a time directly:
  inspect_element (Elements tab, one element's HTML/styles/box),
  get_console_logs (Console tab), get_network_requests (Network
  tab), or switch_devtools_tab (pass tab="elements"/"console"/
  "network"/"sources" to open any one panel, matching how a human
  switches tabs in a real browser's "Inspect Element" devtools
  window). Prefer diagnose_page_issue first since it gathers all of
  this at once; use these individually only when you need to re-check
  one specific panel (e.g. re-reading console logs after retrying an
  action).

Once you have completed the user's task (or if you get
stuck and cannot proceed further), call take_screenshot
as your final action to provide visual proof of the
resulting page before giving your final answer.
"""



def _make_tool_node_and_llm(tool_list):
    return ToolNode(tool_list), llm.bind_tools(tool_list)


def _build_graph(tool_list):
    tool_node, llm_with_tools = _make_tool_node_and_llm(tool_list)

    def guardrail_node(state: AgentState):
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        task_text = last_human.content if last_human else ""

        is_safe, reason = check_request_safety(llm, task_text)

        if is_safe:
            return {"messages": []}

        return {
            "messages": [
                AIMessage(
                    content=f"Request blocked by security guardrail: {reason}",
                    name="guardrail_block",
                )
            ]
        }

    def guardrail_passed(state: AgentState):
        last_message = state["messages"][-1] if state["messages"] else None

        if isinstance(last_message, AIMessage) and last_message.name == "guardrail_block":
            return END

        return "agent"

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def has_taken_screenshot(state: AgentState) -> bool:
        return any(
            isinstance(m, ToolMessage) and m.name == "take_screenshot"
            for m in state["messages"]
        )

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]

        if last_message.tool_calls:
            return "tools"

        if not has_taken_screenshot(state):
            return "screenshot"

        return END

    async def final_screenshot(state: AgentState, config):
        result = await take_screenshot.ainvoke({}, config=config)

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    name="take_screenshot",
                    tool_call_id="final-screenshot"
                )
            ]
        }

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("guardrail", guardrail_node)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("screenshot", final_screenshot)

    graph_builder.add_edge(START, "guardrail")

    graph_builder.add_conditional_edges(
        "guardrail",
        guardrail_passed,
        {
            "agent": "agent",
            END: END,
        }
    )

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "screenshot": "screenshot",
            END: END,
        }
    )

    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("screenshot", "agent")

    return graph_builder.compile(checkpointer=MemorySaver())


WORKER_TOOLS = [
    open_page,
    observe_page,
    type_text,
    click_element,
    press_key,
    select_option,
    set_checkbox,
    fill_date,
    check_form_requirements,
    submit_form,
    scroll_page,
    go_back,
    go_forward,
    close_popup,
    extract_text,
    wait,
    take_screenshot,
    request_human_input,
    grant_media_permissions,
    solve_captcha,
    list_frames_and_canvases,
    click_at_coordinates,
    download_file,
    inspect_element,
    get_console_logs,
    get_network_requests,
    switch_devtools_tab,
    diagnose_page_issue,
]

worker_graph = _build_graph(WORKER_TOOLS)


async def run_task(
    graph,
    task: str,
    thread_id: str,
    session_id: str = DEFAULT_SESSION_ID,
    label: str = "",
    is_subagent: bool = False,
) -> str:
    """Drive `graph` to completion on `task`, resolving human-input
    interrupts via blocking stdin reads (off the event loop), and return
    the final assistant message content.

    `session_id` pins every tool call made during this run to the same
    browser tab (see tools.py's `_session_id` for why this must go
    through config rather than an ambient contextvar).
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
            "session_id": session_id,
        },
        "run_name": label or f"worker: {task[:60]}",
        "tags": ["subagent"] if is_subagent else ["worker"],
        "metadata": {"task": task, "session_id": session_id},
    }

    messages = [
        SystemMessage(content=_worker_system_prompt()),
        HumanMessage(content=task),
    ]

    result = await graph.ainvoke({"messages": messages}, config=config)

    while "__interrupt__" in result:
        interrupt_request = result["__interrupt__"][0]
        question = interrupt_request.value.get("question", "Input needed")
        input_type = interrupt_request.value.get("input_type", "text")

        prefix = f"[{label}] " if label else ""
        prompt = f"\n{prefix}[HUMAN INPUT NEEDED - {input_type}]\n{question}\n> "

        human_reply = await asyncio.to_thread(input, prompt)

        result = await graph.ainvoke(Command(resume=human_reply), config=config)

    return result["messages"][-1].content


async def _run_subagent_task(task: str, index: int) -> dict:
    session_id = await new_session()

    try:
        result = await run_task(
            worker_graph,
            task,
            thread_id=f"subagent-{uuid.uuid4().hex}",
            session_id=session_id,
            label=f"sub-agent {index + 1}: {task[:40]}",
            is_subagent=True,
        )

        return {"task": task, "success": True, "result": result}

    except Exception as e:
        return {"task": task, "success": False, "error": str(e)}

    finally:
        await close_session(session_id)


async def run_parallel_tasks(tasks: List[str]) -> List[dict]:
    """Run several independent tasks concurrently, each in its own
    worker agent and its own isolated browser tab.

    This is the entry point for genuine multi-task parallelism: the
    caller (a human or calling code) decides the tasks are independent,
    so there is no LLM in the loop guessing whether steps depend on each
    other. Use this directly instead of asking a single agent to split
    its own work - a model can't reliably tell "independent tasks" apart
    from "sequential steps of one task" (e.g. search -> size -> address
    -> payment in a checkout flow are NOT independent and must never be
    parallelized).
    """

    return await asyncio.gather(
        *(_run_subagent_task(task, i) for i, task in enumerate(tasks))
    )


# Exposed as a tool only for a caller that explicitly builds an
# orchestrator agent around it (e.g. bind_tools([spawn_subagents]) on its
# own graph). Not included in WORKER_TOOLS or the default `graph` below,
# since letting a single-task agent decide on its own when to fan out
# into sub-agents is exactly what caused one ordering task (login, size,
# address, payment - all sequential) to be split into several broken,
# tab-spawning parallel sub-agents.
@tool
async def spawn_subagents(tasks: List[str]) -> List[dict]:
    """Delegate multiple genuinely independent tasks to parallel browser
    sub-agents. Each task runs to completion in its own browser tab.

    Only call this for tasks that do not depend on each other's outcome
    and could be done in any order or at the same time (e.g. "check the
    price of X on site A" and "check the price of X on site B"). Never
    call this to split the steps of a single task (e.g. do not turn
    "search, pick a size, enter an address, then pay" into several
    sub-agent calls - those steps share one browser session/cart and
    must run in order, in a single agent).

    Args:
        tasks: One self-contained instruction per sub-agent, each with
            enough detail (site, goal, constraints) to be completed
            without further input from you.

    Returns:
        A list of {"task", "success", "result"|"error"} dicts, one per
        input task, in the same order.
    """

    return await run_parallel_tasks(tasks)


graph = worker_graph


async def main():
    task = "open langchain docs switch to dark mode and scrape the first page information."

    print(await run_task(graph, task, thread_id="browser-automation-session"))


if __name__ == "__main__":
    asyncio.run(main())

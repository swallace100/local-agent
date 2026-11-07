"""Graph behavior, tested with a scripted model so no GPU is needed."""

import time
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent.config import Settings
from agent.graph import (
    TOOL_OUTPUT_REMOVED,
    build_graph,
    estimate_tokens,
    fit_context,
    initial_state,
    read_urls,
    unread_citations,
)


class ScriptedModel(BaseChatModel):
    """Plays back a fixed list of replies, and repeats the last one when it runs out."""

    replies: list
    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        # A fresh copy each time, because LangGraph treats a reused message ID as an update.
        return ChatResult(generations=[ChatGeneration(message=reply.model_copy())])


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@tool
def web_read_page(url: str) -> str:
    """Read a page."""
    return f"URL: {url}\nPage text"


@tool
def flaky() -> str:
    """Always fails."""
    raise ValueError("broken")


SETTINGS = Settings()


async def run(replies, tools=(web_read_page,), settings=SETTINGS, task="Do it"):
    model = ScriptedModel(replies=replies, calls=[])
    graph = build_graph(model, list(tools), settings)
    state = await graph.ainvoke(initial_state(task), {"recursion_limit": 200})
    return state, model


async def test_clean_run_finishes_as_done():
    state, _ = await run(
        [
            AIMessage("1. Read"),
            call("web_read_page", {"url": "https://a.org/x"}, "1"),
            AIMessage("Answer from https://a.org/x"),
            AIMessage("PASS"),
        ]
    )
    assert state["status"] == "done"
    assert state["report"] == "Answer from https://a.org/x"
    assert state["revisions"] == 0


async def test_citing_an_unread_url_forces_a_revision():
    state, _ = await run(
        [
            AIMessage("plan"),
            AIMessage("Answer from https://never-opened.org/page"),
            call("web_read_page", {"url": "https://never-opened.org/page"}, "1"),
            AIMessage("Answer from https://never-opened.org/page"),
            AIMessage("PASS"),
        ]
    )
    assert state["status"] == "done"
    assert state["revisions"] == 1
    feedback = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    assert "never-opened.org" in feedback.content


async def test_gives_up_with_warning_after_max_revisions():
    state, _ = await run([AIMessage("plan"), AIMessage("See https://fake.org/a")])
    assert state["status"] == "done_with_warnings"
    assert state["revisions"] == SETTINGS.max_revisions
    assert "fake.org" in state["warnings"][0]


async def test_reviewer_failure_sends_feedback_to_the_agent():
    state, _ = await run(
        [
            AIMessage("plan"),
            AIMessage("First attempt"),
            AIMessage("FAIL\nIt doesn't answer the question"),
            AIMessage("Second attempt"),
            AIMessage("PASS"),
        ]
    )
    assert state["status"] == "done"
    assert state["report"] == "Second attempt"
    assert any("doesn't answer" in str(m.content) for m in state["messages"])


async def test_step_limit_stops_a_looping_agent_and_keeps_a_result():
    settings = replace(SETTINGS, max_steps=3)
    looping = [AIMessage("plan")] + [
        call("web_read_page", {"url": f"https://a.org/{i}"}, str(i)) for i in range(3)
    ] + [AIMessage("Partial result")]
    state, _ = await run(looping, settings=settings)
    assert state["status"] == "budget_exhausted"
    assert state["report"] == "Partial result"
    assert state["steps"] == 3


async def test_agent_is_warned_as_steps_run_out_and_warning_is_not_saved():
    settings = replace(SETTINGS, max_steps=10)
    looping = [AIMessage("plan")] + [
        call("web_read_page", {"url": f"https://a.org/{i}"}, str(i)) for i in range(8)
    ] + [AIMessage("Done"), AIMessage("PASS")]
    state, model = await run(looping, settings=settings)
    sent = ["\n".join(str(m.content) for m in messages) for messages in model.calls]
    assert any("5 of 10 steps" in text and "Cover what's still missing" in text for text in sent)
    assert any("8 of 10 steps" in text and "write your final result now" in text for text in sent)
    assert not any("of 10 steps" in str(m.content) for m in state["messages"])


async def test_tool_errors_go_back_to_the_model_instead_of_crashing():
    state, _ = await run(
        [AIMessage("plan"), call("flaky", {}, "1"), AIMessage("It failed, so here is why"), AIMessage("PASS")],
        tools=(flaky,),
    )
    assert state["status"] == "done"
    tool_result = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_result.status == "error"


async def test_time_limit_stops_the_run():
    settings = replace(SETTINGS, max_minutes=1)
    model = ScriptedModel(
        replies=[AIMessage("plan"), call("web_read_page", {"url": "https://a.org"}, "1"), AIMessage("Late")],
        calls=[],
    )
    graph = build_graph(model, [web_read_page], settings)
    start = initial_state("Do it") | {"started_at": time.time() - 3600}
    state = await graph.ainvoke(start)
    assert state["status"] == "budget_exhausted"


def test_failed_page_reads_dont_count_as_read():
    messages = [
        call("web_read_page", {"url": "https://bad.org"}, "1"),
        ToolMessage("Could not read", tool_call_id="1", name="web_read_page", status="error"),
    ]
    assert read_urls(messages) == set()
    assert unread_citations("See https://bad.org", "task", messages) == ["https://bad.org"]


def test_redirect_target_counts_as_read():
    messages = [
        call("web_read_page", {"url": "https://short.org"}, "1"),
        ToolMessage("URL: https://real.org/page\ntext", tool_call_id="1", name="web_read_page"),
    ]
    assert unread_citations("See https://real.org/page.", "task", messages) == []


def test_urls_given_in_the_task_can_be_cited():
    assert unread_citations("See https://given.org/a", "Summarize https://given.org/a", []) == []


def test_fit_context_blanks_oldest_tool_output_and_keeps_pairs():
    messages = [HumanMessage("task")]
    for i in range(6):
        messages.append(call("web_read_page", {"url": "u"}, str(i)))
        messages.append(ToolMessage("x" * 7000, tool_call_id=str(i), name="web_read_page"))
    fitted = fit_context(messages, max_tokens=7000)
    outputs = [m for m in fitted if isinstance(m, ToolMessage)]
    assert len(fitted) == len(messages)
    assert outputs[0].content == TOOL_OUTPUT_REMOVED
    assert outputs[-1].content == "x" * 7000
    assert estimate_tokens(fitted) <= 7000
    assert messages[2].content == "x" * 7000, "the saved state must not change"

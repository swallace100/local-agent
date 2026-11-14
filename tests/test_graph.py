"""Graph behavior, tested with a scripted model so no GPU is needed."""

import time
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agent.config import Settings
from agent.graph import (
    CHARS_PER_TOKEN,
    TOOL_OUTPUT_REMOVED,
    build_graph,
    calibrate,
    longer_file_written,
    estimate_tokens,
    fit_context,
    initial_state,
    read_urls,
    strip_narration,
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


async def test_reading_pages_but_citing_none_forces_a_revision():
    state, _ = await run(
        [
            AIMessage("plan"),
            call("web_read_page", {"url": "https://a.org/x"}, "1"),
            AIMessage("The answer is 42."),
            AIMessage("The answer is 42 (https://a.org/x)."),
            AIMessage("PASS"),
        ]
    )
    assert state["status"] == "done"
    assert state["revisions"] == 1
    assert state["report"] == "The answer is 42 (https://a.org/x)."


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


def test_urls_containing_brackets_are_matched_whole():
    url = "https://en.wikipedia.org/wiki/Ruby_(programming_language)"
    messages = [
        call("web_read_page", {"url": url}, "1"),
        ToolMessage(f"URL: {url}\ntext", tool_call_id="1", name="web_read_page"),
    ]
    assert unread_citations(f"Ruby dates from 1995 (Source: {url}).", "task", messages) == []
    assert unread_citations("See (https://unread.org/a).", "task", messages) == ["https://unread.org/a"]


def test_a_placeholder_host_is_exempt_only_as_the_host():
    assert unread_citations("See https://docs.example.com/a and http://localhost:8000", "task", []) == []
    hidden = "https://evil.site/example.com/page"
    assert unread_citations(f"See {hidden}", "task", []) == [hidden]
    assert unread_citations("See https://notexample.com/a", "task", []) == ["https://notexample.com/a"]


@tool
def finish_with_file(path: str) -> str:
    """Hand in a file."""
    if path != "result.md":
        raise ValueError("No such file")
    return "# Full document\n\nEvery line of it."


async def test_a_file_handed_in_becomes_the_report():
    state, model = await run(
        [AIMessage("plan"), call("finish_with_file", {"path": "result.md"}, "1"), AIMessage("PASS")],
        tools=(finish_with_file,),
    )
    assert state["status"] == "done"
    assert state["report"] == "# Full document\n\nEvery line of it."
    assert "Every line of it." in model.calls[-1][-1].content, "the reviewer reads the file itself"


async def test_handing_in_a_missing_file_keeps_the_agent_working():
    state, _ = await run(
        [
            AIMessage("plan"),
            call("finish_with_file", {"path": "missing.md"}, "1"),
            AIMessage("The answer, in full"),
            AIMessage("PASS"),
        ],
        tools=(finish_with_file,),
    )
    assert state["status"] == "done"
    assert state["report"] == "The answer, in full"


async def test_describing_a_saved_file_instead_of_handing_it_in_forces_a_revision():
    @tool
    def write_file(path: str, content: str) -> str:
        """Write."""
        return "Wrote it"

    document = "# Full document\n\nEvery line of it."
    replies = [
        AIMessage("plan"),
        call("write_file", {"path": "result.md", "content": "x" * 500}, "1"),
        AIMessage("I've written the quiz you asked for."),
        call("finish_with_file", {"path": "result.md"}, "2"),
        AIMessage("PASS"),
    ]
    model = ScriptedModel(replies=replies, calls=[])
    graph = build_graph(model, [write_file, finish_with_file], SETTINGS, files_are_the_result=True)
    state = await graph.ainvoke(initial_state("Write a quiz"), {"recursion_limit": 200})
    assert state["status"] == "done"
    assert state["revisions"] == 1
    assert state["report"] == document
    assert any("finish_with_file" in str(m.content) for m in state["messages"] if isinstance(m, HumanMessage))


async def test_a_file_that_was_only_working_notes_is_asked_about_once():
    @tool
    def write_file(path: str, content: str) -> str:
        """Write."""
        return "Wrote it"

    replies = [
        AIMessage("plan"),
        call("write_file", {"path": "scratch.md", "content": "x" * 500}, "1"),
        AIMessage("The answer is 42."),
        AIMessage("The answer is 42."),
        AIMessage("PASS"),
    ]
    model = ScriptedModel(replies=replies, calls=[])
    graph = build_graph(model, [write_file, finish_with_file], SETTINGS, files_are_the_result=True)
    state = await graph.ainvoke(initial_state("What is the answer?"), {"recursion_limit": 200})
    assert state["status"] == "done" and state["revisions"] == 1
    assert state["report"] == "The answer is 42."


def test_only_a_file_longer_than_the_reply_raises_the_question():
    wrote = [
        call("write_file", {"path": "a.md", "content": "x" * 100}, "1"),
        call("write_file", {"path": "notes/b.md", "content": "x" * 900}, "2"),
    ]
    assert longer_file_written(wrote, "I've created the quiz.") == "notes/b.md"
    assert longer_file_written(wrote, "The full answer. " * 100) is None
    assert longer_file_written([], "Short") is None


async def test_reviewer_is_told_which_files_were_written():
    @tool
    def write_file(path: str, content: str) -> str:
        """Write."""
        return "Wrote it"

    _, model = await run(
        [AIMessage("plan"), call("write_file", {"path": "notes.md", "content": "x"}, "1"), AIMessage("Done"), AIMessage("PASS")],
        tools=(write_file,),
    )
    assert "Files written: notes.md" in model.calls[-1][-1].content


def test_token_estimate_adjusts_to_the_models_own_count():
    # The first request sets the fixed overhead of tool definitions and the chat template.
    assert calibrate(3_500, 2_000, CHARS_PER_TOKEN, None) == {"token_overhead": 1_000}
    # Dense text: 6,000 characters took 3,000 tokens beyond the overhead.
    assert calibrate(6_000, 4_000, CHARS_PER_TOKEN, 1_000) == {"chars_per_token": 2.0}
    assert calibrate(60_000, 2_000, CHARS_PER_TOKEN, 1_000) == {"chars_per_token": 4.5}, "capped"
    assert calibrate(600, 1_100, CHARS_PER_TOKEN, 1_000) == {}, "too little text to judge"


async def test_measured_token_counts_are_saved_in_the_run():
    def counted(message, tokens):
        usage = {"input_tokens": tokens, "output_tokens": 1, "total_tokens": tokens + 1}
        return message.model_copy(update={"usage_metadata": usage})

    state, _ = await run(
        [
            AIMessage("plan"),
            counted(call("web_read_page", {"url": "https://a.org/x"}, "1"), 1_200),
            counted(AIMessage("Answer from https://a.org/x"), 2_400),
            AIMessage("PASS"),
        ]
    )
    assert state["token_overhead"] > 0
    assert 1.5 <= state["chars_per_token"] <= 4.5


def test_denser_text_makes_the_context_fill_sooner():
    messages = [HumanMessage("task")]
    for i in range(6):
        messages.append(call("web_read_page", {"url": "u"}, str(i)))
        messages.append(ToolMessage("x" * 7000, tool_call_id=str(i), name="web_read_page"))
    blanked = lambda fitted: sum(m.content == TOOL_OUTPUT_REMOVED for m in fitted)  # noqa: E731
    assert blanked(fit_context(messages, 13_000)) == 0
    assert blanked(fit_context(messages, 13_000, chars_per_token=2.0)) == 3
    assert blanked(fit_context(messages, 13_000, overhead=4_000)) > 0


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


def test_opening_narration_is_removed():
    report = "Now I have read 5 posts. Let me compile the report.\n\n## Themes\n\nMemory"
    assert strip_narration(report) == "## Themes\n\nMemory"


def test_a_rule_after_the_narration_is_removed_too():
    assert strip_narration("Let me give the result:\n\n---\n\n**Title**\n\nBody") == "**Title**\n\nBody"


def test_a_normal_opening_is_kept():
    report = "SQLite suits small apps.\n\nDuckDB suits analytics."
    assert strip_narration(report) == report


def test_a_heading_mentioning_narration_words_is_kept():
    report = "# Let me explain\n\nBody"
    assert strip_narration(report) == report


def test_a_report_that_is_only_one_paragraph_is_kept():
    report = "Let me be clear: the answer is 4."
    assert strip_narration(report) == report

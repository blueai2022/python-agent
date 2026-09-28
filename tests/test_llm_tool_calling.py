import pytest
from pydantic import BaseModel

from icd10_coding_agent.llm_tool_calling import (
    BackendError,
    ChatBackend,
    Message,
    Tool,
    ToolCall,
    ToolExecutionError,
    execute_tool_calls,
)


class SearchArgs(BaseModel):
    query: str
    k: int = 5


def test_message_to_openai():
    assert Message(role="user", content="hi").to_openai() == {
        "role": "user",
        "content": "hi",
    }

    assistant = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={"query": "x"})],
    )
    payload = assistant.to_openai()
    assert payload["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'
    assert payload["tool_calls"][0]["function"]["name"] == "search"

    tool = Message(role="tool", tool_call_id="c1", content="result")
    assert tool.to_openai() == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "c1",
    }


def test_tool_definition_schema():
    tool = Tool(name="search", description="search the index", args_model=SearchArgs, func=lambda a: "ok")
    definition = tool.definition()
    assert definition.name == "search"
    assert definition.description == "search the index"
    assert "query" in definition.parameters["properties"]
    assert definition.to_openai()["type"] == "function"


def test_tool_execute_valid():
    def func(args):
        return f"searched {args.query}"

    tool = Tool(name="search", description="d", args_model=SearchArgs, func=func)
    assert tool.execute({"query": "x"}) == "searched x"


def test_tool_execute_invalid_arguments():
    tool = Tool(name="search", description="d", args_model=SearchArgs, func=lambda a: "ok")
    with pytest.raises(ToolExecutionError):
        tool.execute({"k": 3})


def test_execute_tool_calls_success_and_correlation():
    tool = Tool(name="search", description="d", args_model=SearchArgs, func=lambda a: "RESULT")
    assistant = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "x"})],
    )
    replies = execute_tool_calls(assistant, {"search": tool})
    assert len(replies) == 1
    assert replies[0].role == "tool"
    assert replies[0].tool_call_id == "call_1"
    assert replies[0].content == "RESULT"


def test_execute_tool_calls_unknown_tool():
    assistant = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="nope", arguments={})],
    )
    replies = execute_tool_calls(assistant, {})
    assert replies[0].role == "tool"
    assert replies[0].tool_call_id == "c1"
    assert "unknown tool" in replies[0].content


def test_execute_tool_calls_invalid_arguments_reported_as_result():
    tool = Tool(name="search", description="d", args_model=SearchArgs, func=lambda a: "ok")
    assistant = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={})],
    )
    replies = execute_tool_calls(assistant, {"search": tool})
    assert replies[0].role == "tool"
    assert "Error" in replies[0].content


class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, choices):
        self.choices = choices


def test_chat_backend_complete_parses_tool_calls(monkeypatch):
    received = {}

    class FakeCompletions:
        def create(self, **kwargs):
            received.update(kwargs)
            return FakeResponse(
                [
                    FakeChoice(
                        FakeMessage(
                            None,
                            [FakeToolCall("call_1", "search", '{"query": "x"}')],
                        )
                    )
                ]
            )

    def fake_openai(base_url=None, api_key=None):
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": FakeCompletions()})()
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    backend = ChatBackend(model="gpt-4o-mini", base_url="http://h", api_key="k")
    out = backend.complete([Message(role="user", content="hi")])

    assert out.role == "assistant"
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"query": "x"}
    assert received["model"] == "gpt-4o-mini"


def test_chat_backend_transport_failure(monkeypatch):
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("backend unreachable")

    def fake_openai(base_url=None, api_key=None):
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": FailingCompletions()})()
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    backend = ChatBackend(model="m")
    with pytest.raises(BackendError):
        backend.complete([Message(role="user", content="hi")])


def test_chat_backend_placeholder_key_when_none(monkeypatch):
    captured = {}

    class CaptureCompletions:
        def create(self, **kwargs):
            return FakeResponse([FakeChoice(FakeMessage("ok"))])

    def fake_openai(base_url=None, api_key=None):
        captured["api_key"] = api_key
        client = type("Client", (), {})()
        client.chat = type("Chat", (), {"completions": CaptureCompletions()})()
        return client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    ChatBackend(model="m").complete([Message(role="user", content="hi")])
    assert captured["api_key"] == "sk-local"

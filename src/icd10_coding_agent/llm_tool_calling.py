"""Exchange messages and tool calls with an OpenAI-compatible chat backend."""

from __future__ import annotations

import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError


class BackendError(Exception):
    """A transport/backend failure; this aborts the exchange."""


class ToolExecutionError(Exception):
    """A tool-execution failure (e.g. invalid arguments); reported to the model."""


class ToolCall(BaseModel):
    """A single tool-call requested by the backend."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A role-tagged conversation message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in self.tool_calls
            ]
        if self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
        return payload


class ToolDefinition(BaseModel):
    """A tool's name, description, and parameter JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool:
    """An advertised, callable tool backed by a Pydantic arguments model."""

    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        func,
    ) -> None:
        self.name = name
        self.description = description
        self.args_model = args_model
        self.func = func

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    def execute(self, arguments: dict[str, Any]) -> str:
        try:
            args = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"invalid arguments for tool {self.name!r}: {exc}"
            ) from exc
        try:
            return self.func(args)
        except Exception as exc:
            raise ToolExecutionError(f"tool {self.name!r} failed: {exc}") from exc


def execute_tool_calls(assistant: Message, tools: Mapping[str, Tool]) -> list[Message]:
    """Execute every tool call in an assistant message, returning tool-role replies.

    Tool-execution failures (including invalid arguments and unknown tools) are
    reported back as ordinary tool results so the conversation can continue.
    """
    replies: list[Message] = []
    for tool_call in assistant.tool_calls:
        tool = tools.get(tool_call.name)
        if tool is None:
            content = f"Error: unknown tool {tool_call.name!r}"
        else:
            try:
                content = tool.execute(tool_call.arguments)
            except ToolExecutionError as exc:
                content = f"Error: {exc}"
        replies.append(
            Message(role="tool", tool_call_id=tool_call.id, content=content)
        )
    return replies


class ChatBackend:
    """A chat-completions backend configurable by base URL, API key, and model."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
    ) -> Message:
        """Send a conversation and return the next assistant message."""
        from openai import OpenAI

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_openai() for message in messages],
        }
        if tools:
            payload["tools"] = [tool.definition().to_openai() for tool in tools]
        try:
            # Local servers don't require a real key, but the SDK rejects None.
            client = OpenAI(base_url=self.base_url, api_key=self.api_key or "sk-local")
            response = client.chat.completions.create(**payload)
        except Exception as exc:
            raise BackendError(f"chat backend failure: {exc}") from exc
        return _assistant_message_from_openai(response.choices[0].message)


def _assistant_message_from_openai(message: Any) -> Message:
    tool_calls: list[ToolCall] = []
    for raw in message.tool_calls or []:
        arguments: dict[str, Any] = {}
        raw_arguments = raw.function.arguments
        if raw_arguments:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        tool_calls.append(
            ToolCall(id=raw.id, name=raw.function.name, arguments=arguments)
        )
    return Message(role="assistant", content=message.content, tool_calls=tool_calls)

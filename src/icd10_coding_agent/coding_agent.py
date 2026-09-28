"""Iterative, retrieval-augmented reasoning to decide a final ICD-10 code."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ._embeddings import Embedder
from .llm_tool_calling import ChatBackend, Message, Tool, execute_tool_calls
from .semantic_retrieval import retrieve

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_RETRIES = 2


class AgentError(Exception):
    """Raised when the agent cannot produce a final answer."""


class FinalAnswer(BaseModel):
    """The structured final decision: a code (possibly empty) and a reason."""

    code: str = ""
    reason: str = ""


class CaseInput(BaseModel):
    """Input for a single coding decision."""

    evidence: str
    upstream_candidate_code: str


class SearchArgs(BaseModel):
    query: str
    k: int = 5


DEFAULT_SYSTEM_PROMPT = (
    "You are an ICD-10 medical coding assistant. You decide the final ICD-10 code for a "
    "single case, given quoted clinical evidence and an upstream candidate code.\n"
    "You may call the `search` tool any number of times to retrieve candidate codes and "
    "their descriptions from a semantic index; call it whenever you need more evidence.\n"
    "Selection rules:\n"
    "- Choose a final code only from candidates surfaced by `search`; never invent a code "
    "from memory.\n"
    "- Treat the upstream candidate code as a prior to verify, not a certainty to keep "
    "unconditionally.\n"
    "- Prefer a diagnosis/condition code over a symptom/finding code when both plausibly "
    "fit, except that a complication of a diagnosis is itself a condition.\n"
    "- Match the specificity of the evidence: select a specific complication/qualifier "
    "code when the evidence supports it, and an unspecified code only when it does not.\n"
    "When you are ready to conclude, respond with a single JSON object and nothing else, "
    'of the form {"code": "<ICD-10 code, or empty string if none fits>", "reason": '
    '"<brief explanation>"}.\n'
)


class CodingAgent:
    """Decides a final code for a single case via bounded, retrieval-augmented reasoning."""

    def __init__(
        self,
        backend: ChatBackend,
        embeddings: Embedder,
        index_path: str | Path,
        *,
        system_prompt: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.backend = backend
        self.embeddings = embeddings
        self.index_path = Path(index_path)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.messages: list[Message] = []

    def decide(self, case: CaseInput) -> FinalAnswer:
        """Run the reasoning loop and return the final structured decision."""
        self.messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self._user_message(case)),
        ]
        tool = self._search_tool()
        tools: dict[str, Tool] = {tool.name: tool}

        steps = 0
        retries = 0
        while steps < self.max_steps:
            steps += 1
            assistant = self.backend.complete(self.messages, [tool])
            self.messages.append(assistant)

            if assistant.tool_calls:
                self.messages.extend(execute_tool_calls(assistant, tools))
                continue

            answer = self._parse_answer(assistant.content)
            if answer is None:
                retries += 1
                if retries > self.max_retries:
                    raise AgentError("model failed to produce a valid final answer")
                self.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your response was not a valid JSON object with 'code' and "
                            "'reason' fields. Retry with a schema-conformant answer."
                        ),
                    )
                )
                continue

            if answer.code == "" and not answer.reason:
                retries += 1
                if retries > self.max_retries:
                    raise AgentError(
                        "model produced an empty final code without justification"
                    )
                self.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your final code was empty. Reconsider the retrieved "
                            "candidates before concluding: provide a code, or a reason "
                            "explaining why none fit."
                        ),
                    )
                )
                continue

            return answer

        raise AgentError(f"reasoning did not converge within {self.max_steps} steps")

    def _search_tool(self) -> Tool:
        def run_search(args: SearchArgs) -> str:
            results = retrieve(args.query, args.k, self.embeddings, self.index_path)
            return json.dumps(
                [
                    {
                        "code": result.code,
                        "display_text": result.display_text,
                        "cross_references": list(result.cross_references),
                        "score": round(result.score, 4),
                    }
                    for result in results
                ]
            )

        return Tool(
            name="search",
            description="Search the semantic index for candidate ICD-10 codes.",
            args_model=SearchArgs,
            func=run_search,
        )

    @staticmethod
    def _user_message(case: CaseInput) -> str:
        return (
            f"Clinical evidence:\n{case.evidence}\n\n"
            f"Upstream candidate code: {case.upstream_candidate_code}\n\n"
            "Determine the final ICD-10 code for this case."
        )

    @staticmethod
    def _parse_answer(content: str | None) -> FinalAnswer | None:
        if content is None:
            return None
        try:
            return FinalAnswer.model_validate_json(content)
        except ValidationError:
            return None

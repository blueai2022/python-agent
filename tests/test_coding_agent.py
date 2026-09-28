import pytest

from icd10_coding_agent.coding_agent import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYSTEM_PROMPT,
    AgentError,
    CaseInput,
    CodingAgent,
    FinalAnswer,
)
from icd10_coding_agent.llm_tool_calling import Message, ToolCall
from icd10_coding_agent.semantic_retrieval import RetrievalResult


class FakeEmbedder:
    model = "test-model"

    def embed(self, texts):
        return [[0.0] for _ in texts]


class ScriptedBackend:
    def __init__(self, responses):
        self.model = "test-model"
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        return self.responses[idx]


def _answer(code, reason):
    import json

    return Message(role="assistant", content=json.dumps({"code": code, "reason": reason}))


def _search_tool_call(query):
    return Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="search", arguments={"query": query})],
    )


def test_decide_without_lookup():
    backend = ScriptedBackend([_answer("I10", "matches")])
    agent = CodingAgent(backend, FakeEmbedder(), "index.db")
    answer = agent.decide(CaseInput(evidence="htn", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert answer.reason == "matches"
    assert backend.calls == 1


def test_decide_after_multiple_lookups(monkeypatch):
    calls = []

    def fake_retrieve(query, k, embeddings, index_path):
        calls.append((query, k))
        return [RetrievalResult(code="E11.9", display_text="diabetes", cross_references=(), score=0.9)]

    monkeypatch.setattr("icd10_coding_agent.coding_agent.retrieve", fake_retrieve)

    backend = ScriptedBackend(
        [
            _search_tool_call("diabetes complication"),
            _answer("E11.9", "found"),
        ]
    )
    agent = CodingAgent(backend, FakeEmbedder(), "index.db")
    answer = agent.decide(CaseInput(evidence="...", upstream_candidate_code="E11"))
    assert answer.code == "E11.9"
    assert calls == [("diabetes complication", 5)]


def test_decide_never_converging(monkeypatch):
    monkeypatch.setattr(
        "icd10_coding_agent.coding_agent.retrieve",
        lambda q, k, e, i: "[]",
    )

    class AlwaysTool:
        model = "test-model"

        def complete(self, messages, tools=None):
            return _search_tool_call("x")

    agent = CodingAgent(AlwaysTool(), FakeEmbedder(), "index.db", max_steps=3)
    with pytest.raises(AgentError):
        agent.decide(CaseInput(evidence="...", upstream_candidate_code="I10"))


def test_decide_malformed_then_valid():
    backend = ScriptedBackend(
        [Message(role="assistant", content="not json"), _answer("I10", "ok")]
    )
    agent = CodingAgent(backend, FakeEmbedder(), "index.db")
    answer = agent.decide(CaseInput(evidence="...", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert backend.calls == 2


def test_decide_empty_code_then_valid():
    backend = ScriptedBackend([_answer("", ""), _answer("I10", "ok")])
    agent = CodingAgent(backend, FakeEmbedder(), "index.db")
    answer = agent.decide(CaseInput(evidence="...", upstream_candidate_code="I10"))
    assert answer.code == "I10"
    assert backend.calls == 2


def test_decide_empty_code_with_reason_accepted():
    backend = ScriptedBackend([_answer("", "no candidate fits the evidence")])
    agent = CodingAgent(backend, FakeEmbedder(), "index.db")
    answer = agent.decide(CaseInput(evidence="...", upstream_candidate_code="Z99"))
    assert answer.code == ""
    assert answer.reason == "no candidate fits the evidence"
    assert backend.calls == 1


def test_malformed_output_exhausts_retries():
    backend = ScriptedBackend([Message(role="assistant", content="bad")] * 3)
    agent = CodingAgent(backend, FakeEmbedder(), "index.db", max_retries=2)
    with pytest.raises(AgentError):
        agent.decide(CaseInput(evidence="...", upstream_candidate_code="I10"))


def test_default_bounds():
    assert DEFAULT_MAX_STEPS == 10
    assert DEFAULT_MAX_RETRIES == 2


def test_override_bounds():
    agent = CodingAgent(
        ScriptedBackend([]),
        FakeEmbedder(),
        "index.db",
        max_steps=3,
        max_retries=1,
    )
    assert agent.max_steps == 3
    assert agent.max_retries == 1


def test_system_prompt_contains_selection_rules():
    assert "never invent a code" in DEFAULT_SYSTEM_PROMPT
    assert "prior to verify" in DEFAULT_SYSTEM_PROMPT
    assert "Prefer a diagnosis/condition code" in DEFAULT_SYSTEM_PROMPT
    assert "specificity" in DEFAULT_SYSTEM_PROMPT


def test_final_answer_is_pydantic_model():
    answer = FinalAnswer.model_validate_json('{"code": "I10", "reason": "x"}')
    assert answer.code == "I10"
    assert answer.reason == "x"

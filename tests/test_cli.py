from pathlib import Path

from typer.testing import CliRunner

from icd10_coding_agent.cli import Settings, app, resolve_settings
from icd10_coding_agent.coding_agent import FinalAnswer
from icd10_coding_agent.icd10_corpus import Corpus
from icd10_coding_agent.llm_tool_calling import Message

runner = CliRunner()


def test_settings_builtin_defaults():
    settings = Settings()
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.model == "llama3.1:8b"
    assert settings.rag_embed_base_url == "http://localhost:11434/v1"
    assert settings.rag_embed_model == "mxbai-embed-large"
    assert settings.index_path == Path("index.db")
    assert settings.corpus_path == Path("corpus.json")


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_RAG_EMBED_MODEL", "custom-model")
    monkeypatch.setenv("ICD10_AGENT_INDEX_PATH", "/tmp/foo.db")
    settings = Settings()
    assert settings.rag_embed_model == "custom-model"
    assert settings.index_path == Path("/tmp/foo.db")


def test_settings_llm_env_names(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_BASE_URL", "http://h:1234/v1")
    monkeypatch.setenv("ICD10_AGENT_MODEL", "llama3.2")
    settings = Settings()
    assert settings.base_url == "http://h:1234/v1"
    assert settings.model == "llama3.2"


def test_resolve_settings_cli_over_env(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_RAG_EMBED_MODEL", "env-model")
    assert resolve_settings(rag_embed_model="cli-model").rag_embed_model == "cli-model"


def test_resolve_settings_env_over_default(monkeypatch):
    monkeypatch.setenv("ICD10_AGENT_MODEL", "env-llm")
    assert resolve_settings().model == "env-llm"


def test_commands_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index-build" in result.output
    assert "run" in result.output


def test_run_requires_evidence():
    result = runner.invoke(app, ["run"])
    assert result.exit_code != 0


def test_index_build_wires(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]", encoding="utf-8")
    out = tmp_path / "out.db"

    monkeypatch.setattr("icd10_coding_agent.cli.load_corpus", lambda p: Corpus([]))
    monkeypatch.setattr("icd10_coding_agent.cli.load_aliases", lambda p: [])
    monkeypatch.setattr("icd10_coding_agent.cli.validate_aliases", lambda a, c: None)
    captured = {}

    def fake_build(corpus_obj, embeddings, index_path, aliases=()):
        captured["index"] = index_path
        captured["model"] = embeddings.model
        captured["base_url"] = embeddings.base_url

    monkeypatch.setattr("icd10_coding_agent.cli.build_index", fake_build)

    result = runner.invoke(
        app,
        [
            "index-build",
            "--corpus",
            str(corpus),
            "--index",
            str(out),
            "--rag-embed-model",
            "mymodel",
            "--rag-embed-base-url",
            "http://h",
        ],
    )
    assert result.exit_code == 0
    assert captured["index"] == out
    assert captured["model"] == "mymodel"
    assert captured["base_url"] == "http://h"


def test_index_build_wires_aliases(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text("[]", encoding="utf-8")
    aliases_file = tmp_path / "aliases.json"
    aliases_file.write_text("[]", encoding="utf-8")
    out = tmp_path / "out.db"

    loaded_aliases = [object()]

    monkeypatch.setattr("icd10_coding_agent.cli.load_corpus", lambda p: Corpus([]))
    monkeypatch.setattr("icd10_coding_agent.cli.load_aliases", lambda p: loaded_aliases)
    captured = {}

    def fake_validate(aliases, corpus):
        captured["validated_aliases"] = aliases

    def fake_build(corpus_obj, embeddings, index_path, aliases=()):
        captured["aliases"] = aliases

    monkeypatch.setattr("icd10_coding_agent.cli.validate_aliases", fake_validate)
    monkeypatch.setattr("icd10_coding_agent.cli.build_index", fake_build)

    result = runner.invoke(
        app,
        [
            "index-build",
            "--corpus",
            str(corpus),
            "--aliases",
            str(aliases_file),
            "--index",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert captured["validated_aliases"] is loaded_aliases
    assert captured["aliases"] is loaded_aliases


def test_run_verbose_prints_transcript(monkeypatch):
    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.messages = [
                Message(role="system", content="SYS"),
                Message(role="assistant", content=None),
            ]

        def decide(self, case):
            return FinalAnswer(code="I10", reason="x")

    monkeypatch.setattr("icd10_coding_agent.cli.CodingAgent", FakeAgent)

    result = runner.invoke(app, ["run", "the evidence", "I10", "--verbose"])
    assert result.exit_code == 0
    assert "[system] SYS" in result.output
    assert '"code": "I10"' in result.output


def test_run_non_verbose_only_final_decision(monkeypatch):
    class FakeAgent:
        def __init__(self, *args, **kwargs):
            self.messages = [Message(role="system", content="SYS")]

        def decide(self, case):
            return FinalAnswer(code="I10", reason="x")

    monkeypatch.setattr("icd10_coding_agent.cli.CodingAgent", FakeAgent)

    result = runner.invoke(app, ["run", "the evidence", "I10"])
    assert result.exit_code == 0
    assert "[system]" not in result.output
    assert '"code": "I10"' in result.output

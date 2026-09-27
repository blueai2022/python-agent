import csv
import json

import pytest

from icd10_coding_agent.icd10_corpus import (
    CATEGORY_PREFIX_LENGTH,
    ClinicalNotes,
    Corpus,
    CorpusEntry,
    CorpusError,
    load_corpus,
)


def _write_json(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def _write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_entry_fields_are_accessible():
    entry = CorpusEntry(
        code="I10",
        description="Essential (primary) hypertension",
        category="I10",
        billable=True,
        clinical_notes=ClinicalNotes(
            inclusion_terms=("High blood pressure",),
            excludes=(),
            cross_references=(),
        ),
    )
    assert entry.code == "I10"
    assert entry.description == "Essential (primary) hypertension"
    assert entry.category == "I10"
    assert entry.billable is True
    assert entry.clinical_notes.inclusion_terms == ("High blood pressure",)


def test_category_derived_from_code_prefix(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(src, [{"code": "E11.9", "description": "Type 2 diabetes mellitus"}])
    corpus = load_corpus(src)
    assert corpus.lookup("E11.9").category == "E11.9"[:CATEGORY_PREFIX_LENGTH]


def test_explicit_category_preserved(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [{"code": "E11.9", "description": "Type 2 diabetes mellitus", "category": "diabetes"}],
    )
    corpus = load_corpus(src)
    assert corpus.lookup("E11.9").category == "diabetes"


def test_hierarchical_source_keeps_only_leaf_entries(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [
            {"code": "E11", "description": "Type 2 diabetes mellitus", "billable": False},
            {
                "code": "E11.9",
                "description": "Type 2 diabetes without complications",
                "parent": "E11",
            },
            {
                "code": "E11.21",
                "description": "Type 2 diabetes with nephropathy",
                "parent": "E11",
            },
        ],
    )
    corpus = load_corpus(src)
    assert {entry.code for entry in corpus} == {"E11.9", "E11.21"}
    assert corpus.lookup("E11") is None


def test_parent_node_excluded_by_children(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [
            {"code": "E11", "description": "Type 2 diabetes mellitus"},
            {
                "code": "E11.9",
                "description": "Type 2 diabetes without complications",
                "parent": "E11",
            },
        ],
    )
    corpus = load_corpus(src)
    assert {entry.code for entry in corpus} == {"E11.9"}


def test_flat_source_treats_every_entry_as_leaf(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [
            {"code": "I10", "description": "Essential hypertension"},
            {"code": "K21.0", "description": "GERD with esophagitis"},
        ],
    )
    corpus = load_corpus(src)
    assert len(corpus) == 2
    assert all(entry.billable for entry in corpus)


def test_lookup_known_code_case_insensitive(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(src, [{"code": "I10", "description": "Essential hypertension"}])
    corpus = load_corpus(src)
    entry = corpus.lookup("i10")
    assert entry is not None
    assert entry.code == "I10"


def test_lookup_unknown_code_returns_none(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(src, [{"code": "I10", "description": "Essential hypertension"}])
    corpus = load_corpus(src)
    assert corpus.lookup("Z99.9") is None


def test_load_json_maps_all_fields(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [
            {
                "code": "I10",
                "description": "Essential hypertension",
                "category": "hypertension",
                "billable": True,
                "inclusion_terms": ["high blood pressure", "HTN"],
                "excludes": ["secondary hypertension"],
                "cross_references": ["I15"],
            }
        ],
    )
    entry = load_corpus(src).lookup("I10")
    assert entry.category == "hypertension"
    assert entry.billable is True
    assert entry.clinical_notes.inclusion_terms == ("high blood pressure", "HTN")
    assert entry.clinical_notes.excludes == ("secondary hypertension",)
    assert entry.clinical_notes.cross_references == ("I15",)


def test_load_csv_maps_columns(tmp_path):
    src = tmp_path / "corpus.csv"
    _write_csv(
        src,
        ["code", "description"],
        [["I10", "Essential hypertension"], ["K21.0", "GERD with esophagitis"]],
    )
    corpus = load_corpus(src)
    assert {entry.code for entry in corpus} == {"I10", "K21.0"}
    assert corpus.lookup("K21.0").description == "GERD with esophagitis"


def test_missing_code_raises_identifying_record(tmp_path):
    src = tmp_path / "corpus.json"
    _write_json(
        src,
        [
            {"code": "I10", "description": "Essential hypertension"},
            {"description": "no code here"},
        ],
    )
    with pytest.raises(CorpusError) as exc:
        load_corpus(src)
    assert "record 2" in str(exc.value)
    assert "code" in str(exc.value)


def test_missing_description_raises_identifying_record(tmp_path):
    src = tmp_path / "corpus.csv"
    _write_csv(src, ["code", "description"], [["I10", ""]])
    with pytest.raises(CorpusError) as exc:
        load_corpus(src)
    assert "record 1" in str(exc.value)
    assert "description" in str(exc.value)

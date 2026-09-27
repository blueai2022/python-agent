import json

import pytest

from icd10_coding_agent.code_aliasing import (
    Alias,
    AliasError,
    SearchableAlias,
    UnknownAliasTargetError,
    load_aliases,
    validate_aliases,
)
from icd10_coding_agent.icd10_corpus import Corpus, CorpusEntry


def _corpus():
    return Corpus(
        [
            CorpusEntry(
                code="I10",
                description="Essential (primary) hypertension",
                category="I10",
                billable=True,
            ),
            CorpusEntry(
                code="E11.9",
                description="Type 2 diabetes without complications",
                category="E11",
                billable=True,
            ),
        ]
    )


def test_alias_holds_all_three_fields():
    alias = Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")
    assert alias.short_form == "HTN"
    assert alias.target_code == "I10"
    assert alias.expanded_name == "Hypertension"


def test_alias_display_text():
    alias = Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")
    assert alias.display_text == "Hypertension - HTN"


def test_alias_projects_to_searchable_entry():
    alias = Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")
    entry = SearchableAlias.from_alias(alias)
    assert entry.match_text == "HTN"
    assert entry.target_code == "I10"
    assert entry.display_text == "Hypertension - HTN"


def test_validate_known_target_passes():
    aliases = [Alias(short_form="HTN", target_code="I10", expanded_name="Hypertension")]
    validate_aliases(aliases, _corpus())


def test_validate_unknown_target_raises():
    aliases = [Alias(short_form="XXX", target_code="Z99.9", expanded_name="Not real")]
    with pytest.raises(UnknownAliasTargetError) as exc:
        validate_aliases(aliases, _corpus())
    assert "XXX" in str(exc.value)
    assert "Z99.9" in str(exc.value)


def test_load_aliases_json(tmp_path):
    src = tmp_path / "aliases.json"
    src.write_text(
        json.dumps(
            [
                {"acronym": "HTN", "code": "I10", "name": "Hypertension"},
                {"acronym": "DM", "code": "E11.9", "name": "Diabetes mellitus"},
            ]
        ),
        encoding="utf-8",
    )
    aliases = load_aliases(src)
    assert [alias.short_form for alias in aliases] == ["HTN", "DM"]
    assert aliases[0].target_code == "I10"
    assert aliases[0].expanded_name == "Hypertension"


def test_load_aliases_csv(tmp_path):
    src = tmp_path / "aliases.csv"
    src.write_text(
        "acronym,code,name\n"
        "HTN,I10,Hypertension\n"
        "DM,E11.9,Diabetes mellitus\n",
        encoding="utf-8",
    )
    aliases = load_aliases(src)
    assert [(a.short_form, a.target_code, a.expanded_name) for a in aliases] == [
        ("HTN", "I10", "Hypertension"),
        ("DM", "E11.9", "Diabetes mellitus"),
    ]


def test_load_aliases_missing_field_raises(tmp_path):
    src = tmp_path / "aliases.json"
    src.write_text(json.dumps([{"acronym": "HTN", "code": "I10"}]), encoding="utf-8")
    with pytest.raises(AliasError) as exc:
        load_aliases(src)
    assert "name" in str(exc.value)

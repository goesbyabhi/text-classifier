import hashlib
import re

from skill import DESCRIPTION, NAME, SCHEMA_URI, SKILL, digest, index


def test_name_is_a_valid_agent_skill_name():
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", NAME)
    assert 1 <= len(NAME) <= 64


def test_description_under_1024_and_matches_frontmatter():
    assert 0 < len(DESCRIPTION) <= 1024
    assert f"description: {DESCRIPTION}" in SKILL


def test_skill_has_frontmatter_name():
    assert f"name: {NAME}" in SKILL


def test_digest_is_sha256_of_exact_served_bytes():
    expected = "sha256:" + hashlib.sha256(SKILL.encode("utf-8")).hexdigest()
    assert digest() == expected
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest())


def test_skill_is_self_contained_markdown():
    assert SKILL.startswith("---")
    assert "text-classifier" in SKILL
    assert "POST `/v1/classify`" in SKILL


def test_index_shape_and_schema():
    idx = index()
    assert set(idx) == {"$schema", "skills"}
    assert idx["$schema"] == SCHEMA_URI == "https://schemas.agentskills.io/discovery/0.2.0/schema.json"


def test_index_single_skill_entry():
    idx = index()
    assert len(idx["skills"]) == 1
    (entry,) = idx["skills"]
    assert entry["name"] == NAME
    assert entry["type"] == "skill-md"
    assert entry["description"] == DESCRIPTION
    assert entry["url"] == "/skill.md"
    assert entry["digest"] == digest()
"""The licence audit of the bundled third-party set (#465 PR-C4).

docs/reference/native_runtime_resolved_config.md §20. Pins
``scripts/native/license_audit.py`` on the committed audit and on synthetic
wheels:

* the committed ``shipping/license_audit.json`` names exactly the objects of
  ``shipping/third_party_set.json`` and ``shipping/THIRD_PARTY.md`` is its
  rendering;
* coverage rules: a missing, extra or re-assigned object, an unknown status or
  source, a non-clean status without a risk, and a status that contradicts the
  shipped text all fail;
* each object's claims are checked against its own wheel's licence files: two
  wheels whose files are byte-identical are still read one object at a time,
  so a claim that holds for one object does not carry over to the other;
* the official-terms check refuses a snapshot whose sha256 is not the recorded
  one, a missing clause and a release whose version is not the wheel's;
* a check without its inputs is reported as skipped, never as passed.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "scripts/native"


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(NATIVE))
    spec = importlib.util.spec_from_file_location(name, NATIVE / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


la = _load("license_audit")
SET = json.loads((REPO / "shipping/third_party_set.json").read_text())
AUDIT = json.loads((REPO / "shipping/license_audit.json").read_text())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# the committed audit


def test_committed_audit_covers_the_set_and_renders_the_notice() -> None:
    assert la.check_coverage(SET, AUDIT) == []
    assert (REPO / "shipping/THIRD_PARTY.md").read_text() == la.render(SET, AUDIT)
    assert AUDIT["distribution"]["status"] == "local-only"
    assert AUDIT["distribution"]["owner_confirmation"] is None


def test_committed_audit_statuses() -> None:
    by = {o["soname"]: o["status"] for o in AUDIT["objects"]}
    for s in ("libnvJitLink.so.13", "libcufile.so.0", "libnvshmem_host.so.3"):
        assert by[s] == "grant_in_official_only"
    assert by["libgomp.so.1"] == "no_licence_text_shipped"
    assert by["libnvinfer.so.10"] == "grant_in_both_texts_differ"


# ---------------------------------------------------------------------------
# coverage


def _drop(a: dict) -> None:
    a["objects"].pop()


def _extra(a: dict) -> None:
    a["objects"].append({**a["objects"][0], "soname": "libextra.so.1"})


def _rewheel(a: dict) -> None:
    a["objects"][0]["wheel"] = "other-1.0"


def _status(a: dict) -> None:
    a["objects"][0]["status"] = "fine"


def _source(a: dict) -> None:
    a["objects"][0]["official"][0]["source"] = "nowhere"


def _norisk(a: dict) -> None:
    o = next(o for o in a["objects"] if o["status"] == "grant_in_official_only")
    o["risks"] = []


def _silent_but_clean(a: dict) -> None:
    o = next(o for o in a["objects"] if o["status"] == "grant_in_official_only")
    o["status"] = "grant_in_bundled_and_official"


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_drop, "in the third-party set but not in the audit"),
        (_extra, "in the audit but not in the third-party set"),
        (_rewheel, "!= the set's"),
        (_status, "unknown status"),
        (_source, "unknown source"),
        (_norisk, "without a risk"),
        (_silent_but_clean, "shipped text silent but status"),
    ],
)
def test_coverage_rejects(edit, problem: str) -> None:
    a = copy.deepcopy(AUDIT)
    edit(a)
    assert any(problem in p for p in la.check_coverage(SET, a)), la.check_coverage(
        SET, a
    )


# ---------------------------------------------------------------------------
# claims, object by object, on synthetic wheels

EULA = (
    "Agreement text. Your application must have material additional functionality.\n"
    "2.6. Attachment A\nThe following portions of the SDK are distributable under the Agreement:\n"
    "Component\nLinux\nlibalpha.so, libalpha_static.a\n2.7. Attachment B\nthird party\n"
)


def _synthetic(tmp: Path, listed_beta: bool) -> tuple[dict, dict, Path]:
    """Two wheels carrying byte-identical License.txt files that list only
    libalpha.so; the audit claims libbeta.so is `listed_beta`."""
    lic = tmp / "licenses"
    entries = []
    for soname, wheel in (("libalpha.so.1", "alpha-1.0"), ("libbeta.so.1", "beta-1.0")):
        (lic / wheel).mkdir(parents=True)
        (lic / wheel / "License.txt").write_text(EULA)
        entries.append(
            {
                "soname": soname,
                "wheel": wheel,
                "source": {"root": "purelib", "path": f"{wheel}/{soname}"},
                "license_files": [
                    {
                        "path": f"{wheel}.dist-info/License.txt",
                        "sha256": _sha(EULA.encode()),
                    }
                ],
            }
        )
    set_ = {"entries": entries}

    def obj(soname: str, wheel: str, name: str, listed: bool) -> dict[str, Any]:
        return {
            "soname": soname,
            "wheel": wheel,
            "release": {"name": "r", "source": "src"},
            "bundled": {"attachment_a": {"name": name, "listed": listed}},
            "official": [
                {"source": "src", "attachment_a": {"name": name, "listed": True}}
            ],
            "conditions": "sdk",
            "status": "grant_in_bundled_and_official"
            if listed
            else "grant_in_official_only",
            "risks": [] if listed else ["silent"],
        }

    audit = {
        "schema": la.AUDIT_SCHEMA,
        "distribution": {
            "status": "local-only",
            "owner_confirmation": None,
            "reading": "r",
        },
        "statuses": dict(AUDIT["statuses"]),
        "conditions": {
            "sdk": ["Your application must have material additional functionality."]
        },
        "sources": {
            "src": {
                "url": "https://example.invalid/r/v1.0/",
                "snapshot": "src.txt",
                "sha256": "",
                "version": "v",
                "fetched": "t",
            }
        },
        "objects": [
            obj("libalpha.so.1", "alpha-1.0", "libalpha.so", True),
            obj("libbeta.so.1", "beta-1.0", "libbeta.so", listed_beta),
        ],
    }
    return set_, audit, lic


def test_identical_files_are_still_read_per_object(tmp_path: Path) -> None:
    set_, audit, lic = _synthetic(tmp_path, listed_beta=False)
    shipped, bad = la.shipped_files(set_, lic, None)
    assert bad == []
    assert la.check_bundled(audit, shipped) == []

    _, audit_wrong, _ = _synthetic(tmp_path / "w", listed_beta=True)
    problems = la.check_bundled(audit_wrong, shipped)
    assert problems == [
        "libbeta.so.1 (shipped): Attachment A does not list 'libbeta.so'"
    ]


def test_shipped_file_with_other_bytes_fails(tmp_path: Path) -> None:
    set_, _, lic = _synthetic(tmp_path, listed_beta=False)
    (lic / "beta-1.0/License.txt").write_text(EULA + "changed\n")
    _, bad = la.shipped_files(set_, lic, None)
    assert len(bad) == 1 and "beta-1.0" in bad[0]


def test_condition_must_be_in_the_objects_own_text(tmp_path: Path) -> None:
    set_, audit, lic = _synthetic(tmp_path, listed_beta=False)
    audit["conditions"]["sdk"].append("A condition no file states.")
    shipped, _ = la.shipped_files(set_, lic, None)
    assert len(la.check_bundled(audit, shipped)) == 2


def test_official_terms_snapshot_clause_and_release(tmp_path: Path) -> None:
    set_, audit, lic = _synthetic(tmp_path, listed_beta=False)
    src = tmp_path / "sources"
    src.mkdir()
    text = EULA.replace("libalpha.so, libalpha_static.a", "libalpha.so, libbeta.so")
    (src / "src.txt").write_text(text)
    audit["sources"]["src"]["sha256"] = _sha(text.encode())
    shipped, _ = la.shipped_files(set_, lic, None)
    bad, notes = la.check_official(audit, src, shipped)
    assert bad == []
    # the URL's tag v1.0 is the wheel's version 1.0
    assert notes == []

    a = copy.deepcopy(audit)
    a["sources"]["src"]["sha256"] = "0" * 64
    assert any("is not sha256" in p for p in la.check_official(a, src, shipped)[0])

    a = copy.deepcopy(audit)
    a["objects"][1]["official"][0]["clause"] = "a clause the terms do not have"
    assert any("clause not found" in p for p in la.check_official(a, src, shipped)[0])

    a = copy.deepcopy(audit)
    a["sources"]["src"]["url"] = "https://example.invalid/r/latest/"
    assert len(la.check_official(a, src, shipped)[1]) == 2


def test_switcher_version_must_be_the_wheels(tmp_path: Path) -> None:
    set_, audit, lic = _synthetic(tmp_path, listed_beta=False)
    src = tmp_path / "sources"
    src.mkdir()
    page = (
        "<html><script>DOCUMENTATION_OPTIONS.theme_switcher_version_match = '2.0';</script>"
        "<body>" + EULA.replace("libalpha_static.a", "libbeta.so") + "</body></html>"
    ).encode()
    (src / "src.txt").write_bytes(page)
    audit["sources"]["src"]["sha256"] = _sha(page)
    bad, _ = la.check_official(audit, src, None)
    assert [p for p in bad if "is version 2.0" in p] and len(bad) == 2


def test_check_without_inputs_is_skipped_not_passed(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    assert la.main(["check", "--report", str(report)]) == 0
    r = json.loads(report.read_text())
    assert r["checks"]["bundled_texts"] == {"skipped": "no --licenses / --purelib"}
    assert r["checks"]["official_terms"] == {"skipped": "no --sources"}
    assert r["complete"] is False and r["pass"] is True
    assert r["distribution"] == "local-only"


def test_render_check_detects_a_hand_edit(tmp_path: Path) -> None:
    out = tmp_path / "THIRD_PARTY.md"
    assert la.main(["render", "--out", str(out)]) == 0
    assert la.main(["render", "--out", str(out), "--check"]) == 0
    out.write_text(out.read_text() + "edit\n")
    assert la.main(["render", "--out", str(out), "--check"]) == 1

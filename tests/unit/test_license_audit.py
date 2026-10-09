"""The licence audit of the bundled third-party set (#465 PR-C4, #547).

docs/reference/native_runtime_resolved_config.md §20, §22. Pins
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
* a check without its inputs is reported as skipped, never as passed;
* #547: every open item keeps technical evidence, licence interpretation and
  legal uncertainty as separate non-empty lists, and while one is OPEN the
  distribution stays local-only with no owner confirmation; the downstream
  terms stay a draft whose clauses apply only to objects under NVIDIA's terms
  and whose quotes are found in each cited object's own wheel text; supplied
  texts carry their recorded bytes, official ones are re-derived from their
  snapshots, and the libgomp correspondence check fails on a changed section,
  a missing RPATH rewrite or a COPYING text that is not the source package's.
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
    assert by["libgomp.so.1"] == "grant_in_supplied_text"
    assert by["libnvinfer.so.10"] == "grant_in_both_texts_differ"


def test_committed_open_items_stay_open_and_local_only() -> None:
    """#547: no item is closed and nothing is confirmed by this change."""
    assert [it["id"] for it in AUDIT["open_items"]] == [
        "L-1",
        "L-2",
        "L-3",
        "L-4",
        "M-1",
    ]
    assert all(it["status"] == "OPEN" for it in AUDIT["open_items"])
    assert AUDIT["downstream_terms"]["status"] == "draft"
    assert (
        REPO / "shipping/DOWNSTREAM_TERMS.draft.md"
    ).read_text() == la.render_downstream(AUDIT)
    assert la.check_supplied(AUDIT, la.supplied_bytes(AUDIT), "repo") == []


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
    bad, notes = la.check_official(a, src, shipped)
    # unversioned: a note where a risk is recorded (libbeta), a failure where not
    assert notes == [
        "libbeta.so.1: release src not matched to 1.0 (unversioned; risk recorded)"
    ]
    assert bad == ["libalpha.so.1: release src is not tied to version 1.0"]


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
    assert r["checks"]["downstream_basis"] == {"skipped": "no --licenses / --purelib"}
    assert r["checks"]["corresponding_source"] == {"skipped": "no --gomp-rpm / --srpm"}
    assert r["checks"]["supplied_texts"]["pass"] is True
    assert r["complete"] is False and r["pass"] is True
    assert r["distribution"] == "local-only"


def test_render_check_detects_a_hand_edit(tmp_path: Path) -> None:
    out, down = tmp_path / "THIRD_PARTY.md", tmp_path / "DOWNSTREAM.md"
    args = ["render", "--out", str(out), "--downstream", str(down)]
    assert la.main(args) == 0
    assert la.main([*args, "--check"]) == 0
    for f in (out, down):
        before = f.read_text()
        f.write_text(before + "edit\n")
        assert la.main([*args, "--check"]) == 1
        f.write_text(before)


# ---------------------------------------------------------------------------
# review (2026-10-08): release rows, terms bound to the release, grant evidence

RELNOTES = (
    "Table 1 CUDA 13.0 Update 2 Component Versions Component Name Version "
    "CUDA Runtime (cudart) 13.0.96 x86_64 CUDA NVRTC 13.0.88 x86_64 CUDA nvJitLink 13.0.88 x86_64 "
    "Windows CUPTI 13.0.85 x86_64 Table 2 driver"
).encode()


def _bound_audit(wheel: str, terms_url: str) -> tuple[dict, dict, dict[str, bytes]]:
    obj = {
        "soname": "libx.so",
        "wheel": wheel,
        "release": {"name": "CUDA 13.0 Update 2", "source": "rel"},
        "official": [
            {"source": "terms", "attachment_a": {"name": "libx.so", "listed": True}}
        ],
        "risks": [],
    }
    audit = {
        "sources": {
            "rel": {
                "url": "https://docs.example/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html"
            },
            "terms": {"url": terms_url},
        }
    }
    return obj, audit, {"rel": RELNOTES, "terms": b"terms"}


@pytest.mark.parametrize(
    ("wheel", "ok"),
    [
        ("nvidia_cuda_runtime-13.0.96", True),
        # 13.0.88 is on the page (NVRTC, nvJitLink) but not in cudart's row
        ("nvidia_cuda_runtime-13.0.88", False),
        ("nvidia_cuda_cupti-13.0.85", True),
        ("nvidia_unknown-13.0.96", False),
    ],
)
def test_release_notes_match_the_components_row(wheel: str, ok: bool) -> None:
    obj, audit, snap = _bound_audit(
        wheel, "https://docs.example/cuda/archive/13.0.2/eula/index.html"
    )
    assert la.source_bound(obj, "rel", audit, snap, release=True)[0] is ok


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("https://docs.example/cuda/archive/13.0.2/eula/index.html", True),
        ("https://docs.example/cuda/archive/13.0.1/eula/index.html", False),
        # current, unversioned terms are not the object's release's
        ("https://docs.example/cuda/eula/index.html", False),
    ],
)
def test_terms_are_bound_to_the_validated_release(url: str, ok: bool) -> None:
    obj, audit, snap = _bound_audit("nvidia_cuda_runtime-13.0.96", url)
    assert la.source_bound(obj, "terms", audit, snap, release=False)[0] is ok
    # a release that does not match makes its archive's terms unbound too
    obj, audit, snap = _bound_audit("nvidia_cuda_runtime-13.0.88", url)
    assert la.source_bound(obj, "terms", audit, snap, release=False)[0] is False


def _no_grant(a: dict) -> None:
    o = next(o for o in a["objects"] if o["soname"] == "libnvJitLink.so.13")
    o["official"] = []


def _only_not_listed(a: dict) -> None:
    o = next(o for o in a["objects"] if o["soname"] == "libnvJitLink.so.13")
    o["official"] = [
        {
            "source": "cuda_eula_13.0.2",
            "attachment_a": {"name": "libnvJitLink.so", "listed": False},
        }
    ]


def _granting_but_official_only(a: dict) -> None:
    o = next(o for o in a["objects"] if o["soname"] == "libcudart.so.13")
    o["status"] = "grant_in_official_only"


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_no_grant, "without an official claim that grants distribution"),
        (_only_not_listed, "without an official claim that grants distribution"),
        (_granting_but_official_only, "shipped text grants but status"),
    ],
)
def test_status_needs_matching_grant_evidence(edit, problem: str) -> None:
    a = copy.deepcopy(AUDIT)
    edit(a)
    assert any(problem in p for p in la.check_coverage(SET, a)), la.check_coverage(
        SET, a
    )


# ---------------------------------------------------------------------------
# #547: open items, downstream terms, supplied texts


def _item(a: dict, i: str) -> dict:
    return next(it for it in a["open_items"] if it["id"] == i)


def _empty_layer(a: dict) -> None:
    _item(a, "L-3")["legal_uncertainty"] = []


def _missing_layer(a: dict) -> None:
    del _item(a, "L-1")["licence_interpretation"]


def _public_while_open(a: dict) -> None:
    a["distribution"]["status"] = "public"


def _confirmed_while_open(a: dict) -> None:
    a["distribution"]["owner_confirmation"] = "2026-10-08"


def _item_status(a: dict) -> None:
    _item(a, "L-2")["status"] = "DONE"


def _item_unknown_object(a: dict) -> None:
    _item(a, "L-1")["objects"].append("libnone.so")


def _terms_adopted(a: dict) -> None:
    a["downstream_terms"]["status"] = "in force"


def _clause_on_gpl(a: dict) -> None:
    a["downstream_terms"]["clauses"][0]["applies_to"] = {"objects": ["libgomp.so.1"]}


def _clause_quotes_short(a: dict) -> None:
    c = a["downstream_terms"]["clauses"][1]
    c["basis"] = [
        b for b in c["basis"] if b["applies_to"] != {"conditions": ["tensorrt_wheel"]}
    ]


def _supplied_unknown(a: dict) -> None:
    o = next(o for o in a["objects"] if o["soname"] == "libcufile.so.0")
    o["supplied"][0]["file"] = "licenses/terms/none.txt"


def _supplied_outside(a: dict) -> None:
    a["supplied_texts"][0]["file"] = "lib/vendor/terms.txt"


def _supplied_sha(a: dict) -> None:
    a["supplied_texts"][0]["sha256"] = "0" * 64


def _gpl_without_record(a: dict) -> None:
    a["corresponding_source"] = {}


def _supplied_status_without_text(a: dict) -> None:
    o = next(o for o in a["objects"] if o["soname"] == "libgomp.so.1")
    o["supplied"] = []


def _cuda_text(a: dict) -> dict:
    return next(
        t for t in a["supplied_texts"] if t["file"].endswith("cuda_eula_13.0.2.txt")
    )


def _repo_file_escapes(a: dict) -> None:
    _cuda_text(a)["repo_file"] = "../../outside/cuda_eula_13.0.2.txt"


def _repo_file_elsewhere(a: dict) -> None:
    _cuda_text(a)["repo_file"] = "docs/cuda_eula_13.0.2.txt"


def _file_dotdot(a: dict) -> None:
    t = _cuda_text(a)
    t["file"], t["repo_file"] = "licenses/terms/..", "shipping/licenses/terms/.."


def _close_all(a: dict) -> None:
    for it in a["open_items"]:
        it["status"] = "CLOSED"
        it["owner_conclusion"] = "owner: closed on legal advice"


def _closed_without_conclusion(a: dict) -> None:
    _item(a, "L-1")["status"] = "CLOSED"


def _all_closed_public_unconfirmed(a: dict) -> None:
    _close_all(a)
    a["distribution"]["status"] = "public"


def _all_closed_public_unpublished_source(a: dict) -> None:
    _close_all(a)
    a["distribution"].update(status="public", owner_confirmation="2026-12-01 owner")


def _all_closed_public_draft_terms(a: dict) -> None:
    _all_closed_public_unpublished_source(a)


def test_closing_every_item_does_not_open_distribution() -> None:
    """All items CLOSED (with conclusions) is allowed while local-only; it is
    the distribution status that needs the public gate, not the items."""
    a = copy.deepcopy(AUDIT)
    _close_all(a)
    assert la.check_open_items(a) == []
    assert la.check_public_gate(a) == []
    a["distribution"]["status"] = "public"
    gate = la.check_public_gate(a)
    assert len(gate) == 3 and all(p.startswith("distribution 'public'") for p in gate)
    a["distribution"]["owner_confirmation"] = "2026-12-01 owner"
    a["corresponding_source"]["libgomp.so.1"]["mirror"]["status"] = "published"
    a["downstream_terms"]["status"] = "adopted"
    assert la.check_public_gate(a) == []
    assert la.check_downstream_records(a) == []
    # adopted terms are refused while local-only
    a["distribution"]["status"] = "local-only"
    assert any(
        "only the owner adopts terms" in p for p in la.check_downstream_records(a)
    )


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (_empty_layer, "legal_uncertainty must be a non-empty list"),
        (_missing_layer, "licence_interpretation must be a non-empty list"),
        (_public_while_open, "while items are OPEN"),
        (_confirmed_while_open, "owner_confirmation is set while items are OPEN"),
        (_item_status, "is not OPEN or CLOSED"),
        (_item_unknown_object, "unknown objects"),
        (_terms_adopted, "only the owner adopts terms"),
        (_clause_on_gpl, "outside NVIDIA's terms"),
        (_clause_quotes_short, "quotes cover"),
        (_supplied_unknown, "supplied claim names unknown file"),
        (_supplied_outside, "not under licenses/terms/"),
        (_supplied_sha, "is not sha256"),
        (_gpl_without_record, "GPL object without a corresponding_source record"),
        (
            _supplied_status_without_text,
            "needs an absent wheel text and a supplied text",
        ),
        (_repo_file_escapes, "is not shipping/licenses/terms/cuda_eula_13.0.2.txt"),
        (_repo_file_elsewhere, "is not shipping/licenses/terms/cuda_eula_13.0.2.txt"),
        (_file_dotdot, "not under licenses/terms/"),
        (_closed_without_conclusion, "CLOSED without the owner's conclusion"),
        (_all_closed_public_unconfirmed, "without owner_confirmation"),
        (
            _all_closed_public_unpublished_source,
            "corresponding source of libgomp.so.1 is not published",
        ),
        (_all_closed_public_draft_terms, "downstream terms are not adopted"),
    ],
)
def test_coverage_rejects_547(edit, problem: str) -> None:
    a = copy.deepcopy(AUDIT)
    edit(a)
    bad = la.check_coverage(SET, a)
    assert any(problem in p for p in bad), bad


def test_supplied_claims_are_read_in_the_supplied_text() -> None:
    texts = la.supplied_bytes(AUDIT)
    a = copy.deepcopy(AUDIT)
    o = next(o for o in a["objects"] if o["soname"] == "libnvJitLink.so.13")
    o["supplied"][0]["attachment_a"]["name"] = "libnotthere.so"
    assert la.check_supplied(a, texts, "repo") == [
        "libnvJitLink.so.13 (licenses/terms/cuda_eula_13.0.2.txt): Attachment A does not list 'libnotthere.so'"
    ]
    # libgomp's GPL conditions are checked in the texts the package supplies
    t = dict(texts)
    t["licenses/libgomp/COPYING3"] = b"GNU GPL without section 6"
    bad = la.check_supplied(AUDIT, t, "tree")
    assert any("licenses/libgomp/COPYING3 is not sha256" in p for p in bad)
    assert any("libgomp.so.1 (supplied): condition not found" in p for p in bad)


def test_downstream_quotes_are_read_per_object(tmp_path: Path) -> None:
    shipped = {
        w: [b"Text that quotes nothing."]
        for w in {o["wheel"] for o in AUDIT["objects"]}
    }
    bad = la.check_downstream_basis(AUDIT, shipped)
    cited = {
        so
        for c in AUDIT["downstream_terms"]["clauses"]
        for b in c["basis"]
        for so in la.clause_objects(AUDIT, b)
    }
    assert {p.split("(")[1].split(")")[0] for p in bad} == cited
    assert "libgomp.so.1" not in cited and "libtorch.so" not in cited


PAGE = (
    "<html><head><script>var x = 1;</script></head><body><nav>menu</nav>"
    '<div itemprop="articleBody"><h1>Title<a class="headerlink" href="#t">#</a></h1>'
    "<p>First   clause\n continues.</p><table><tr><td>libalpha.so</td><td>Linux</td></tr></table>"
    "<button>copy</button><ul><li>one</li><li>two</li></ul></div>"
    "<footer>Privacy Policy</footer></body></html>"
)


def test_terms_text_keeps_only_the_article() -> None:
    assert la.terms_text(PAGE.encode()) == (
        "Title\nFirst clause\ncontinues.\nlibalpha.so\nLinux\none\ntwo\n"
    )
    pydata = PAGE.replace(
        '<div itemprop="articleBody">', '<article class="bd-article">'
    ).replace("</div><footer>", "</article><footer>")
    assert la.terms_text(pydata.encode()) == la.terms_text(PAGE.encode())
    with pytest.raises(ValueError):
        la.terms_text(b"<html><body><p>no article</p></body></html>")


def test_supplied_official_text_is_rederived(tmp_path: Path) -> None:
    src = tmp_path / "sources"
    src.mkdir()
    a = copy.deepcopy(AUDIT)
    t = next(
        t for t in a["supplied_texts"] if t["from"].get("source") == "cuda_eula_13.0.2"
    )
    (src / a["sources"]["cuda_eula_13.0.2"]["snapshot"]).write_bytes(PAGE.encode())
    texts = {t["file"]: la.terms_text(PAGE.encode()).encode()}
    a["supplied_texts"] = [t]
    assert la.check_derived_texts(a, src, texts) == []
    texts[t["file"]] += b"an added line\n"
    assert la.check_derived_texts(a, src, texts) == [
        "licenses/terms/cuda_eula_13.0.2.txt: not the article text of cuda_eula_13.0.2"
    ]


def _rpm(files: dict[str, bytes], tags: dict[int, str]) -> bytes:
    """A minimal RPM: lead, empty signature header, a main header with string
    tags, and a gzip newc cpio payload."""
    import gzip
    import struct

    def header(entries: dict[int, str]) -> bytes:
        store = b""
        index = b""
        for tag, val in entries.items():
            index += struct.pack(">IIII", tag, 6, len(store), 1)
            store += val.encode() + b"\0"
        return (
            b"\x8e\xad\xe8\x01\0\0\0\0"
            + struct.pack(">II", len(entries), len(store))
            + index
            + store
        )

    cpio = b""
    for name, data in [*files.items(), ("TRAILER!!!", b"")]:
        n = name.encode() + b"\0"
        mode = 0o100644 if name != "TRAILER!!!" else 0
        fields = [0, mode, 0, 0, 1, 0, len(data), 0, 0, 0, 0, len(n), 0]
        h = b"070701" + b"".join(b"%08X" % f for f in fields) + n
        h += b"\0" * (-len(h) % 4)
        cpio += h + data + b"\0" * (-len(data) % 4)
    sig = header({})
    sig += b"\0" * (-len(sig) % 8)
    lead = b"\xed\xab\xee\xdb" + b"\0" * 92
    return lead + sig + header({**tags, 1125: "gzip"}) + gzip.compress(cpio)


def test_rpm_reader_on_a_synthetic_package() -> None:
    data = _rpm(
        {"./usr/lib64/libx.so.1": b"ELF bytes", "./COPYING3": b"GPL"},
        {1000: "x", 1044: "x-1.src.rpm"},
    )
    tags, _ = la.rpm_header(data)
    assert tags["NAME"] == "x" and tags["SOURCERPM"] == "x-1.src.rpm"
    assert la.rpm_members(data) == {
        "/usr/lib64/libx.so.1": b"ELF bytes",
        "/COPYING3": b"GPL",
    }
    with pytest.raises(ValueError):
        la.rpm_header(b"not an rpm" + data)


GOMP = REPO / "results/547_l3_gomp/20261008/downloads"


@pytest.mark.skipif(
    not (GOMP / "gcc-8.5.0-28.el8_10.alma.1.src.rpm").is_file(),
    reason="needs the #547 source and binary packages (results/, not tracked)",
)
def test_corresponding_source_on_the_real_packages() -> None:
    e = next(e for e in SET["entries"] if e["soname"] == "libgomp.so.1")
    import site

    obj = next(
        (
            Path(p) / e["source"]["path"]
            for p in site.getsitepackages()
            if (Path(p) / e["source"]["path"]).is_file()
        ),
        None,
    )
    if obj is None:
        pytest.skip("the torch wheel's libgomp is not installed")
    data = obj.read_bytes()
    rpm = (GOMP / "libgomp-8.5.0-28.el8_10.alma.1.x86_64.rpm").read_bytes()
    srpm = (GOMP / "gcc-8.5.0-28.el8_10.alma.1.src.rpm").read_bytes()
    texts = la.supplied_bytes(AUDIT)
    assert la.check_corresponding(AUDIT, "libgomp.so.1", data, rpm, srpm, texts) == []
    member = la.rpm_members(rpm)["/usr/lib64/libgomp.so.1.0.0"]
    # the upstream file itself lacks the wheel's RPATH
    assert la.check_corresponding(AUDIT, "libgomp.so.1", member, rpm, srpm, texts) == [
        "libgomp.so.1: .dynamic differs beyond the added DT_RPATH"
    ]
    text = la.elf_sections(data)[".text"]
    flipped = bytearray(data)
    flipped[data.find(text["body"][:64]) + 64] ^= 1
    assert la.check_corresponding(
        AUDIT, "libgomp.so.1", bytes(flipped), rpm, srpm, texts
    ) == ["libgomp.so.1: section .text differs"]
    # a SHT_NOBITS section has no bytes: its size is what must match
    import struct

    shoff = struct.unpack_from("<Q", data, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    stro = struct.unpack_from("<Q", data, shoff + shstrndx * shentsize + 0x18)[0]
    grown = bytearray(data)
    for i in range(shnum):
        h = shoff + i * shentsize
        name = struct.unpack_from("<I", data, h)[0]
        if data[stro + name : stro + name + 5] == b".bss\0":
            size = struct.unpack_from("<Q", data, h + 0x20)[0]
            struct.pack_into("<Q", grown, h + 0x20, size + 64)
    assert la.check_corresponding(
        AUDIT, "libgomp.so.1", bytes(grown), rpm, srpm, texts
    ) == ["libgomp.so.1: section .bss differs"]
    t = dict(texts)
    t["licenses/libgomp/COPYING.RUNTIME"] = b"other"
    assert la.check_corresponding(AUDIT, "libgomp.so.1", data, rpm, srpm, t) == [
        "libgomp.so.1: licenses/libgomp/COPYING.RUNTIME is not gcc-8.5.0-20210514.tar.xz:gcc-8.5.0-20210514/COPYING.RUNTIME"
    ]

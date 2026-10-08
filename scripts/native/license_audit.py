#!/usr/bin/env python3
# status: diagnostic
"""The licence audit of the bundled third-party set (#465 Phase C PR-C4, #547).

``shipping/license_audit.json`` records, per object in
``shipping/third_party_set.json``: the licence text its wheel carries (the
files the package ships under ``licenses/<wheel>/``), the version-matched
official terms (URL, fetch time, sha256 of the snapshot), any licence text the
package supplies itself (``supplied_texts``: official terms under
``licenses/terms/``, GNU libgomp's licence and source directions under
``licenses/libgomp/``), whether each grants distribution of that object (an
Attachment A name, a quoted clause, or a file identical to the shipped one),
the conditions, a status and the open risks. #547 adds ``open_items``
(technical evidence, licence interpretation and legal uncertainty kept apart),
``corresponding_source`` (libgomp's binary and source package) and
``downstream_terms`` (a draft, never in force). It is a reading of licence
texts, not a legal conclusion (docs/reference/native_runtime_resolved_config.md
§20, §22). Every claim is checked against the object's own wheel's files: two
wheels whose files have the same bytes are still checked one by one.

Subcommands:

``render``  write ``shipping/THIRD_PARTY.md`` and
            ``shipping/DOWNSTREAM_TERMS.draft.md`` from the third-party set and
            the audit, or with ``--check`` exit 1 if a committed file is not
            that rendering.
``check``   ``coverage``: the audit names exactly the set's objects, each with
            its wheel, a known status, known sources and conditions; the
            supplied texts, open items (three non-empty layers; while one is
            OPEN the distribution is local-only with no owner confirmation),
            downstream terms (a draft; clauses scoped to NVIDIA objects) and
            corresponding-source records are well formed;
            ``supplied_texts``: the repo files (and with ``--licenses`` the
            tree's) carry the recorded bytes and each object's claims about
            them hold; ``bundled_texts`` (``--licenses DIR``: a tree's
            ``licenses/``, or ``--purelib`` + ``--nvjpeg-wheel``): each shipped
            licence file has the set's sha256, and each object's claims about
            it hold (Attachment A lists / does not list its name, a clause or
            condition appears verbatim up to whitespace, a token is absent);
            ``downstream_basis`` (same inputs): each quote a clause rests on is
            in the text it names; ``official_terms`` (``--sources DIR``, the
            snapshots): each snapshot has the recorded sha256, each object's
            claims about the official terms hold, the release the audit names
            matches the wheel's version (the documentation's version switcher,
            the release notes' component table, or the tag in the URL), and
            each supplied official text is its snapshot's article text,
            re-derived; ``corresponding_source`` (``--gomp-rpm`` + ``--srpm``):
            the packages have the recorded sha256, the binary package's
            SOURCERPM is the source package, the shipped object matches the
            package's file section by section except the recorded rewrite, and
            the supplied COPYING texts are the source package's; ``notice``:
            the committed renderings are current. A check that was not given
            its inputs is reported as ``skipped``, not passed.

Usage::

    license_audit.py render
    license_audit.py check --licenses TREE/licenses --sources SNAPSHOTS \
        --gomp-rpm libgomp-....rpm --srpm gcc-....src.rpm --report audit.json

Exit 0: every check that ran passes; 1: a check fails; 2: error.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import html.parser
import io
import json
import lzma
import re
import struct
import sys
import tarfile
from pathlib import Path
from typing import Any

SCHEMA = "saccade.shipping_license_audit_check/v1"
AUDIT_SCHEMA = "saccade.shipping_license_audit/v2"
REPO = Path(__file__).resolve().parents[2]
THIRD_PARTY_SET = REPO / "shipping/third_party_set.json"
AUDIT = REPO / "shipping/license_audit.json"
NOTICE = REPO / "shipping/THIRD_PARTY.md"
DOWNSTREAM = REPO / "shipping/DOWNSTREAM_TERMS.draft.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def page_text(data: bytes) -> str:
    """The visible text of a snapshot (HTML or plain), whitespace-normalized."""
    t = data.decode("utf-8", errors="replace")
    if "<html" in t[:4096].lower() or "<!doctype html" in t[:4096].lower():
        t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
        t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return normalize(t)


def attachment_a(text: str) -> str | None:
    """The CUDA EULA's Attachment A section (from its heading to Attachment B),
    or None when the text has none. The heading also appears in the table of
    contents of an HTML page, so the longest such span is the section."""
    spans = []
    for m in re.finditer(r"2\.6\.? Attachment A", text):
        end = text.find("Attachment B", m.end())
        spans.append(text[m.start() : end if end > 0 else len(text)])
    return max(spans, key=len) if spans else None


def version_of(wheel: str) -> str:
    return wheel.rsplit("-", 1)[1]


# ------------------------------------------------- supplied texts (#547)

_BLOCK = {
    "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "li",
    "ul", "ol", "table", "thead", "tbody", "tr", "dl", "dt", "dd", "pre",
    "blockquote", "br", "hr", "caption", "td", "th",
}  # fmt: skip
_VOID = {"br", "hr", "img", "meta", "link", "input", "wbr"}
_SKIP = {"script", "style", "nav", "button"}


class _ArticleText(html.parser.HTMLParser):
    """The text of a documentation page's article body: the element with
    itemprop="articleBody" (CUDA docs) or <article class="bd-article"> (the
    pydata theme), without navigation, scripts or heading permalinks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []  # (tag, skipped) inside the body
        self.skip = 0
        self.inside = False
        self.out: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if not self.inside:
            if a.get("itemprop") == "articleBody" or (
                tag == "article" and "bd-article" in (a.get("class") or "")
            ):
                self.inside = True
                self.stack = [(tag, False)]
            return
        if tag in _VOID:
            if tag in _BLOCK:
                self.out.append("\n")
            return
        skipped = tag in _SKIP or "headerlink" in (a.get("class") or "")
        self.skip += skipped
        self.stack.append((tag, skipped))
        if tag in _BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self.inside:
            return
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:  # closes unclosed children too
                self.skip -= sum(s for _, s in self.stack[i:])
                del self.stack[i:]
                break
        else:
            return
        if tag in _BLOCK:
            self.out.append("\n")
        if not self.stack:
            self.inside = False

    def handle_data(self, data: str) -> None:
        if self.inside and not self.skip:
            self.out.append(data)


def terms_text(data: bytes) -> str:
    """The licence text of an official terms page (a snapshot under
    --sources), as the package ships it: the article body, one block per line,
    runs of blanks collapsed, empty lines dropped. Deterministic, so a shipped
    text is checked by re-deriving it from the snapshot."""
    p = _ArticleText()
    p.feed(data.decode("utf-8"))
    p.close()
    lines = (
        re.sub(r"[ \t\r\f\v ]+", " ", line).strip()
        for line in "".join(p.out).split("\n")
    )
    text = "\n".join(line for line in lines if line)
    if not text:
        raise ValueError("no article body in the snapshot")
    return text + "\n"


def rpm_header(data: bytes) -> tuple[dict[str, Any], int]:
    """(selected tags of an RPM's main header, offset of its payload)."""

    def header(off: int) -> tuple[dict[int, Any], int]:
        if data[off : off + 3] != b"\x8e\xad\xe8":
            raise ValueError("not an RPM header")
        nidx, hsize = struct.unpack_from(">II", data, off + 8)
        store = off + 16 + nidx * 16
        tags: dict[int, Any] = {}
        for i in range(nidx):
            tag, typ, o, _ = struct.unpack_from(">IIII", data, off + 16 + i * 16)
            p = store + o
            if typ in (6, 9):  # STRING, I18NSTRING
                tags[tag] = data[p : data.index(b"\0", p)].decode()
            elif typ == 4:  # INT32 (first value)
                tags[tag] = struct.unpack_from(">I", data, p)[0]
        return tags, store + hsize

    if data[:4] != b"\xed\xab\xee\xdb":
        raise ValueError("not an RPM file")
    _, end = header(96)  # the signature header, padded to 8
    tags, payload = header((end + 7) & ~7)
    names = {
        1000: "NAME", 1001: "VERSION", 1002: "RELEASE", 1006: "BUILDTIME",
        1007: "BUILDHOST", 1011: "VENDOR", 1014: "LICENSE", 1044: "SOURCERPM",
        1125: "PAYLOADCOMPRESSOR",
    }  # fmt: skip
    return {v: tags[k] for k, v in names.items() if k in tags}, payload


def rpm_members(data: bytes) -> dict[str, bytes]:
    """name -> bytes of the regular files in an RPM's cpio (newc) payload."""
    tags, off = rpm_header(data)
    comp = tags.get("PAYLOADCOMPRESSOR", "gzip")
    raw = data[off:]
    if comp == "xz":
        raw = lzma.decompress(raw)
    elif comp == "gzip":
        raw = gzip.decompress(raw)
    else:
        raise ValueError(f"unsupported RPM payload compressor {comp!r}")
    out: dict[str, bytes] = {}
    p = 0
    while True:
        if raw[p : p + 6] != b"070701":
            raise ValueError("not a newc cpio payload")
        f = [int(raw[p + 6 + 8 * i : p + 14 + 8 * i], 16) for i in range(13)]
        mode, size, namesize = f[1], f[6], f[11]
        name = raw[p + 110 : p + 110 + namesize - 1].decode()
        p = (p + 110 + namesize + 3) & ~3
        if name == "TRAILER!!!":
            return out
        if mode & 0o170000 == 0o100000:
            out[name.removeprefix(".")] = raw[p : p + size]
        p = (p + size + 3) & ~3


def tar_xz_members(data: bytes, names: set[str]) -> dict[str, bytes]:
    """The named members of an .tar.xz, read as a stream."""
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|xz") as tf:
        for m in tf:
            if m.name in names and m.isfile():
                fh = tf.extractfile(m)
                assert fh is not None
                out[m.name] = fh.read()
    return out


def elf_sections(data: bytes) -> dict[str, dict[str, Any]]:
    """name -> {addr, size, sha256, body} of an ELF64 little-endian object."""
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise ValueError("not an ELF64 little-endian object")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    stro = hdrs[shstrndx][4]
    out = {}
    for name, typ, _, addr, off, size, *_ in hdrs:
        n = data[stro + name : data.index(b"\0", stro + name)].decode()
        body = b"" if typ == 8 else data[off : off + size]  # SHT_NOBITS
        out[n] = {"addr": addr, "size": size, "sha256": _sha256(body), "body": body}
    return out


_DT = {1: "NEEDED", 14: "SONAME", 15: "RPATH", 29: "RUNPATH"}


def elf_dynamic(data: bytes) -> tuple[list[tuple[Any, ...]], list[tuple[str, Any]]]:
    """(.dynsym as (name, info, other, value, size) without section indices,
    .dynamic as (tag, value) with string values resolved)."""
    sec = elf_sections(data)
    strs = sec[".dynstr"]["body"]

    def cstr(o: int) -> str:
        return strs[o : strs.index(b"\0", o)].decode()

    syms = []
    body = sec[".dynsym"]["body"]
    for i in range(0, len(body), 24):
        name, info, other, _shndx, value, size = struct.unpack_from("<IBBHQQ", body, i)
        syms.append((cstr(name), info, other, value, size))
    dyn = []
    body = sec[".dynamic"]["body"]
    for i in range(0, len(body), 16):
        tag, val = struct.unpack_from("<qQ", body, i)
        if tag:
            dyn.append((_DT[tag], cstr(val)) if tag in _DT else (str(tag), val))
    return syms, dyn


# ---------------------------------------------------------------- render


def _claim_summary(claim: dict[str, Any]) -> str:
    if "attachment_a" in claim:
        a = claim["attachment_a"]
        return (
            f"Attachment A {'lists' if a['listed'] else 'does not list'} `{a['name']}`"
        )
    if "clause" in claim:
        return "yes (quoted in the audit)"
    return "no text for this object"


def render(set_: dict[str, Any], audit: dict[str, Any]) -> str:
    entries = {e["soname"]: e for e in set_["entries"]}
    sources = audit["sources"]
    lines = [
        "# Third-party components",
        "",
        "Generated by `scripts/native/license_audit.py render` from "
        "`shipping/third_party_set.json` and `shipping/license_audit.json`; do not edit by hand.",
        "",
        "The shared objects under `lib/vendor/` are third-party components distributed "
        "under their own license terms, copied unmodified from the wheels named below. "
        "They are not covered by Saccade's Apache-2.0 license. Each wheel's license "
        "files are under `licenses/<wheel>/`. License texts the package supplies itself "
        "(version-matched official terms, and GNU libgomp's license and source directions) "
        "are under `licenses/terms/` and `licenses/libgomp/`.",
        "",
        "What follows is a reading of the license texts, not a legal conclusion. Nothing "
        "in this package is an agreement with its recipients.",
        "",
        f"**Distribution status: {audit['distribution']['status']}.** "
        + audit["distribution"]["reading"],
        "",
        "## Objects",
        "",
        "| object | wheel | license files | wheel's text grants distribution "
        "| version-matched official terms | text the package supplies | status |",
        "|:--|:--|:--|:--|:--|:--|:--|",
    ]
    for o in audit["objects"]:
        e = entries[o["soname"]]
        lics = ", ".join(
            f"`{f['path'].rsplit('/', 1)[-1]}`" for f in e["license_files"]
        )
        official = "; ".join(_official_summary(c, sources) for c in o["official"])
        supplied = (
            "; ".join(
                f"`{c['file']}`: {_claim_summary(c)}" for c in o.get("supplied", [])
            )
            or "none"
        )
        lines.append(
            f"| `{o['soname']}` | `{o['wheel']}` | {lics} | {_claim_summary(o['bundled'])} "
            f"| {official} | {supplied} | `{o['status']}` |"
        )
    lines += ["", "Statuses:", ""]
    for k, v in audit["statuses"].items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "## Open items",
        "",
        "Each item keeps apart what was measured or read (technical evidence), how the "
        "license texts read (interpretation) and what only a legal conclusion can settle "
        "(legal uncertainty). While any item is OPEN the distribution status stays "
        "local-only.",
    ]
    for it in audit["open_items"]:
        scope = ", ".join(f"`{s}`" for s in it["objects"]) or it.get("scope", "")
        lines += [
            "",
            f"### {it['id']} ({it['status']}): {it['title']}",
            "",
            f"Scope: {scope}",
            "",
        ]
        for key, label in (
            ("technical_evidence", "Technical evidence"),
            ("licence_interpretation", "License interpretation"),
            ("legal_uncertainty", "Legal uncertainty"),
        ):
            lines.append(f"{label}:")
            lines.append("")
            lines += [f"- {x}" for x in it[key]]
            lines.append("")
        lines.pop()
    lines += ["", "## Notes per object", ""]
    for o in audit["objects"]:
        for r in o["risks"]:
            lines.append(f"- `{o['soname']}`: {r}")
    lines += ["", "## Distribution conditions (quoted)", ""]
    for k, conds in audit["conditions"].items():
        users = sorted({o["wheel"] for o in audit["objects"] if o["conditions"] == k})
        lines.append(f"- {', '.join(f'`{u}`' for u in users)}:")
        for c in conds:
            lines.append(f"  - “{c}”")
    lines += ["", "## Supplied license texts", ""]
    for t in audit["supplied_texts"]:
        f = t["from"]
        if "source" in f:
            origin = (
                f"the article text of `{f['source']}` (re-derived from its snapshot)"
            )
        elif "corresponding_source" in f:
            origin = f"`{f['member']}` in the source package of `{f['corresponding_source']}`"
        else:
            origin = f["authored"]
        lines.append(
            f"- `{t['file']}`: {t['title']}; from {origin} (sha256 `{t['sha256'][:16]}…`)"
        )
    lines += ["", "## Corresponding source", ""]
    for so, cs in audit["corresponding_source"].items():
        b, s = cs["binary_package"], cs["source_package"]
        lines += [
            f"- `{so}`: {cs['reading']}",
            f"  - binary package `{b['name']}` (sha256 `{b['sha256'][:16]}…`), member "
            f"`{b['member']}`; SOURCERPM `{b['sourcerpm']}`",
            f"  - source package `{s['name']}`, sha256 `{s['sha256']}`, {s['bytes']} bytes: "
            + ", ".join(f"<{u}>" for u in s["urls"]),
            f"  - same build ID `{cs['correspondence']['build_id']}`; differing sections: "
            + ", ".join(f"`{k}`" for k in cs["correspondence"]["rewritten_sections"]),
            f"  - publication: {cs['mirror']['status']}. {cs['mirror']['rule']}",
        ]
    lines += [
        "",
        "## Downstream terms",
        "",
        f"A {audit['downstream_terms']['status']} of terms for the NVIDIA components is kept "
        "in the Saccade repository (`shipping/DOWNSTREAM_TERMS.draft.md`). It is not in force "
        "and is not part of this package.",
    ]
    lines += ["", "## Sources", ""]
    for k, s in sources.items():
        lines.append(
            f"- `{k}`: {s['version']}. <{s['url']}> (fetched {s['fetched']}, sha256 `{s['sha256'][:16]}…`)"
        )
    return "\n".join(lines) + "\n"


def clause_objects(audit: dict[str, Any], clause: dict[str, Any]) -> list[str]:
    """The sonames a downstream clause (or one of its quotes) applies to."""
    scope = clause["applies_to"]
    if "objects" in scope:
        return list(scope["objects"])
    return [
        o["soname"] for o in audit["objects"] if o["conditions"] in scope["conditions"]
    ]


def render_downstream(audit: dict[str, Any]) -> str:
    d = audit["downstream_terms"]
    lines = [
        "# Downstream terms for the NVIDIA components (DRAFT)",
        "",
        "Generated by `scripts/native/license_audit.py render` from "
        "`shipping/license_audit.json` (`downstream_terms`); do not edit by hand.",
        "",
        f"**Status: {d['status']}. Not in force.** {d['reading']}",
        "",
        "Open items L-2 and L-4 (`shipping/THIRD_PARTY.md`) say what is unsettled.",
    ]
    for c in d["clauses"]:
        objs = clause_objects(audit, c)
        lines += [
            "",
            f"## {c['id']}",
            "",
            c["text"],
            "",
            "Applies to: " + ", ".join(f"`{o}`" for o in objs),
            "",
            "Rests on:",
            "",
        ]
        for b in c["basis"]:
            scope = b["applies_to"]
            who = (
                ", ".join(f"`{o}`" for o in scope["objects"])
                if "objects" in scope
                else "the objects under "
                + ", ".join(f"`{k}`" for k in scope["conditions"])
            )
            lines.append(f"- in the wheel's text of {who}: “{b['quote']}”")
    return "\n".join(lines) + "\n"


def _official_summary(claim: dict[str, Any], sources: dict[str, Any]) -> str:
    src = f"`{claim['source']}`"
    if "attachment_a" in claim:
        a = claim["attachment_a"]
        return f"{src}: Attachment A {'lists' if a['listed'] else 'does not list'} `{a['name']}`"
    if "identical_to_bundled" in claim:
        return f"{src}: identical to the shipped `{claim['identical_to_bundled']}`"
    if "prefix_of_bundled" in claim:
        return f"{src}: the start of the shipped `{claim['prefix_of_bundled']}`"
    return f"{src}: grants (quoted in the audit)"


# ---------------------------------------------------------------- check


# Statuses that assert a grant in the shipped text / in the official terms.
SHIPPED_GRANT_STATUSES = ("grant_in_bundled_and_official", "grant_in_both_texts_differ")
OFFICIAL_GRANT_STATUSES = (*SHIPPED_GRANT_STATUSES, "grant_in_official_only")
# The row label of each CUDA component wheel in the release notes' "Component
# Versions" table; a release-notes release is matched on that row only.
RELNOTES_COMPONENT = {
    "nvidia_cuda_runtime": "CUDA Runtime (cudart)",
    "nvidia_cublas": "CUDA cuBLAS",
    "nvidia_cufft": "CUDA cuFFT",
    "nvidia_cufile": "CUDA cuFile",
    "nvidia_curand": "CUDA cuRAND",
    "nvidia_cusparse": "CUDA cuSPARSE",
    "nvidia_nvjitlink": "CUDA nvJitLink",
    "nvidia_nvjpeg": "CUDA nvJPEG",
    "nvidia_cuda_nvrtc": "CUDA NVRTC",
    "nvidia_cuda_cupti": "CUPTI",
}


def grants(claim: dict[str, Any]) -> bool:
    """Whether an official claim asserts that the terms grant distribution:
    a listed Attachment A name, a quoted clause, or upstream text identical to
    (or the start of) the shipped grant text. A not-listed name does not."""
    if "attachment_a" in claim:
        return claim["attachment_a"]["listed"] is True
    return any(
        k in claim for k in ("clause", "identical_to_bundled", "prefix_of_bundled")
    )


def component_versions(text: str, label: str) -> list[str]:
    """The versions in `label`'s row of the release notes' component table."""
    start = text.find("Component Versions")
    if start < 0:
        return []
    end = text.find("Table 2", start)
    table = text[start : end if end > 0 else len(text)]
    return re.findall(rf"(?<![\w(]){re.escape(label)} (\d[\d.]*)", table)


def check_coverage(set_: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    bad = []
    if audit.get("schema") != AUDIT_SCHEMA:
        bad.append(f"schema {audit.get('schema')!r} != {AUDIT_SCHEMA}")
    want = {e["soname"]: e["wheel"] for e in set_["entries"]}
    got: dict[str, str] = {}
    supplied_files = {t["file"] for t in audit.get("supplied_texts", [])}
    bad += check_supplied_records(audit)
    bad += check_open_items(audit)
    bad += check_downstream_records(audit)
    bad += check_corresponding_records(audit)
    for o in audit["objects"]:
        if o["soname"] in got:
            bad.append(f"{o['soname']}: listed twice")
        got[o["soname"]] = o["wheel"]
        if o["status"] not in audit["statuses"]:
            bad.append(f"{o['soname']}: unknown status {o['status']!r}")
        if o["conditions"] is None and o["status"] != "no_licence_text_shipped":
            bad.append(f"{o['soname']}: no conditions but status {o['status']}")
        if o["conditions"] is not None and o["conditions"] not in audit["conditions"]:
            bad.append(f"{o['soname']}: unknown conditions {o['conditions']!r}")
        if o["status"] != "grant_in_bundled_and_official" and not o["risks"]:
            bad.append(f"{o['soname']}: status {o['status']} without a risk")
        for c in [o["release"], *o["official"]]:
            if c["source"] not in audit["sources"]:
                bad.append(f"{o['soname']}: unknown source {c['source']!r}")
        b = o["bundled"]
        kinds = [k for k in ("attachment_a", "clause", "absent") if k in b]
        if len(kinds) != 1:
            bad.append(
                f"{o['soname']}: bundled claim must be one of attachment_a/clause/absent, has {kinds}"
            )
        supplied = o.get("supplied", [])
        for c in supplied:
            if c.get("file") not in supplied_files:
                bad.append(
                    f"{o['soname']}: supplied claim names unknown file {c.get('file')!r}"
                )
            if not any(k in c for k in ("attachment_a", "clause")):
                bad.append(
                    f"{o['soname']}: supplied claim needs attachment_a or clause"
                )
        supplied_grants = any(grants(c) for c in supplied)
        if "absent" in b and o["status"] not in (
            "no_licence_text_shipped",
            "grant_in_supplied_text",
        ):
            bad.append(f"{o['soname']}: no shipped text but status {o['status']}")
        if o["status"] == "grant_in_supplied_text" and not (
            "absent" in b and supplied_grants
        ):
            bad.append(
                f"{o['soname']}: status grant_in_supplied_text needs an absent wheel text and a supplied text that grants"
            )
        if o["status"] == "no_licence_text_shipped" and supplied:
            bad.append(
                f"{o['soname']}: status no_licence_text_shipped but the package supplies a text"
            )
        if (
            b.get("attachment_a", {}).get("listed") is False
            and o["status"] != "grant_in_official_only"
        ):
            bad.append(f"{o['soname']}: shipped text silent but status {o['status']}")
        shipped_grants = (
            b.get("attachment_a", {}).get("listed") is True or "clause" in b
        )
        if shipped_grants and o["status"] not in SHIPPED_GRANT_STATUSES:
            bad.append(f"{o['soname']}: shipped text grants but status {o['status']}")
        if o["status"] in OFFICIAL_GRANT_STATUSES and not any(
            grants(c) for c in o["official"]
        ):
            bad.append(
                f"{o['soname']}: status {o['status']} without an official claim that grants distribution"
            )
    for s in sorted(set(want) | set(got)):
        if s not in got:
            bad.append(f"{s}: in the third-party set but not in the audit")
        elif s not in want:
            bad.append(f"{s}: in the audit but not in the third-party set")
        elif want[s] != got[s]:
            bad.append(f"{s}: wheel {got[s]} != the set's {want[s]}")
    return bad


SUPPLIED_DIRS = ("licenses/terms/", "licenses/libgomp/")
OPEN_ITEM_LAYERS = ("technical_evidence", "licence_interpretation", "legal_uncertainty")


def check_supplied_records(audit: dict[str, Any]) -> list[str]:
    """Each supplied text: a package path under licenses/terms/ or
    licenses/libgomp/, a repo file with the recorded sha256, and an origin
    (an official-terms source, a member of a corresponding source package, or
    authored)."""
    bad = []
    seen = set()
    for t in audit.get("supplied_texts", []):
        f = t.get("file", "")
        if f in seen:
            bad.append(f"supplied text {f}: listed twice")
        seen.add(f)
        if not f.startswith(SUPPLIED_DIRS) or ".." in f:
            bad.append(f"supplied text {f}: not under {' or '.join(SUPPLIED_DIRS)}")
        repo_file = REPO / t.get("repo_file", "")
        data = repo_file.read_bytes() if repo_file.is_file() else b""
        if _sha256(data) != t.get("sha256"):
            bad.append(
                f"supplied text {f}: {t.get('repo_file')} is not sha256 {t.get('sha256')}"
            )
        origin = t.get("from", {})
        if "source" in origin:
            if (
                origin["source"] not in audit["sources"]
                or origin.get("method") != "terms_text"
            ):
                bad.append(f"supplied text {f}: unknown source or method {origin}")
        elif "corresponding_source" in origin:
            if origin["corresponding_source"] not in audit.get(
                "corresponding_source", {}
            ):
                bad.append(f"supplied text {f}: unknown corresponding source {origin}")
        elif "authored" not in origin:
            bad.append(f"supplied text {f}: no origin")
    return bad


def check_open_items(audit: dict[str, Any]) -> list[str]:
    """Every open item has the three layers, each non-empty, and names known
    objects; while any item is OPEN the distribution is local-only and has no
    owner confirmation (only the owner closes items, and never this tool)."""
    bad = []
    sonames = {o["soname"] for o in audit["objects"]}
    items = audit.get("open_items", [])
    if not items:
        bad.append("open_items: none recorded")
    ids = [it.get("id") for it in items]
    if len(ids) != len(set(ids)):
        bad.append(f"open_items: duplicate ids {ids}")
    for it in items:
        who = f"open item {it.get('id')}"
        if it.get("status") not in ("OPEN", "CLOSED"):
            bad.append(f"{who}: status {it.get('status')!r} is not OPEN or CLOSED")
        for layer in OPEN_ITEM_LAYERS:
            v = it.get(layer)
            if (
                not isinstance(v, list)
                or not v
                or not all(isinstance(x, str) and x for x in v)
            ):
                bad.append(f"{who}: {layer} must be a non-empty list of statements")
        unknown = [s for s in it.get("objects", []) if s not in sonames]
        if unknown:
            bad.append(f"{who}: unknown objects {unknown}")
        if not it.get("objects") and not it.get("scope"):
            bad.append(f"{who}: names neither objects nor a scope")
    if any(it.get("status") == "OPEN" for it in items):
        dist = audit["distribution"]
        if dist.get("status") != "local-only":
            bad.append(f"distribution {dist.get('status')!r} while items are OPEN")
        if dist.get("owner_confirmation") is not None:
            bad.append("owner_confirmation is set while items are OPEN")
    return bad


def check_downstream_records(audit: dict[str, Any]) -> list[str]:
    """The downstream terms stay a draft; each clause applies to known objects
    (by object or by conditions key) and rests on at least one quote."""
    bad = []
    d = audit.get("downstream_terms")
    if d is None:
        return ["downstream_terms: missing"]
    if d.get("status") != "draft":
        bad.append(
            f"downstream_terms: status {d.get('status')!r}; only the owner adopts terms"
        )
    sonames = {o["soname"] for o in audit["objects"]}
    for c in d.get("clauses", []):
        who = f"downstream {c.get('id')}"
        scope = c.get("applies_to", {})
        key = (
            "objects"
            if "objects" in scope
            else "conditions"
            if "conditions" in scope
            else None
        )
        known = sonames if key == "objects" else set(audit["conditions"])
        if key is None:
            bad.append(f"{who}: applies_to names neither objects nor conditions")
            objs: list[str] = []
        elif not scope[key] or any(x not in known for x in scope[key]):
            bad.append(
                f"{who}: applies_to {key} {scope[key]} names an unknown or no entry"
            )
            objs = []
        else:
            objs = clause_objects(audit, c)
        restricted = [
            o["soname"] for o in audit["objects"]
            if o["soname"] in objs and o["conditions"] not in ("nvidia_sdk", "tensorrt_wheel")
        ]  # fmt: skip
        if restricted:
            bad.append(f"{who}: applies to objects outside NVIDIA's terms {restricted}")
        if not c.get("basis"):
            bad.append(f"{who}: no basis")
        cited: set[str] = set()
        for b in c.get("basis", []):
            bs = b.get("applies_to", {})
            k = (
                "objects"
                if "objects" in bs
                else "conditions"
                if "conditions" in bs
                else None
            )
            ok = (
                k is not None
                and bs[k]
                and all(
                    x in (sonames if k == "objects" else audit["conditions"])
                    for x in bs[k]
                )
            )
            if not ok:
                bad.append(
                    f"{who}: a quote's applies_to is not a known object or conditions list"
                )
                continue
            cited |= set(clause_objects(audit, b))
        if objs and cited != set(objs):
            bad.append(
                f"{who}: quotes cover {sorted(cited)}, the clause applies to {sorted(objs)}"
            )
    return bad


CORRESPONDING_KEYS = (
    "binary_package",
    "source_package",
    "authentication",
    "correspondence",
    "mirror",
)


def check_corresponding_records(audit: dict[str, Any]) -> list[str]:
    bad = []
    sonames = {o["soname"] for o in audit["objects"]}
    for so, cs in audit.get("corresponding_source", {}).items():
        if so not in sonames:
            bad.append(f"corresponding source {so}: not an audited object")
        missing = [k for k in CORRESPONDING_KEYS if k not in cs]
        if missing:
            bad.append(f"corresponding source {so}: missing {missing}")
            continue
        if cs["binary_package"].get("sourcerpm") != cs["source_package"].get("name"):
            bad.append(
                f"corresponding source {so}: SOURCERPM is not the source package"
            )
    for o in audit["objects"]:
        if o["conditions"] == "gpl3_rle" and o["soname"] not in audit.get(
            "corresponding_source", {}
        ):
            bad.append(
                f"{o['soname']}: GPL object without a corresponding_source record"
            )
    return bad


def _claim_problems(
    who: str, claim: dict[str, Any], text: str, conds: list[str] | None = None
) -> list[str]:
    bad = []
    if "attachment_a" in claim:
        a = claim["attachment_a"]
        section = attachment_a(text)
        if section is None:
            bad.append(f"{who}: no Attachment A section")
        elif (a["name"] in section) != a["listed"]:
            bad.append(
                f"{who}: Attachment A {'does not list' if a['listed'] else 'lists'} {a['name']!r}"
            )
    if "clause" in claim and normalize(claim["clause"]) not in text:
        bad.append(f"{who}: clause not found: {claim['clause'][:80]!r}")
    if "absent" in claim and claim["absent"].lower() in text.lower():
        bad.append(f"{who}: {claim['absent']!r} is present")
    for c in conds or []:
        if normalize(c) not in text:
            bad.append(f"{who}: condition not found: {c[:80]!r}")
    return bad


def shipped_files(
    set_: dict[str, Any],
    licenses: Path | None,
    roots: dict[str, Path] | None,
) -> tuple[dict[str, list[bytes]], list[str]]:
    """wheel -> its shipped licence files' bytes, each checked against the set."""
    out: dict[str, list[bytes]] = {}
    bad = []
    for e in set_["entries"]:
        if e["wheel"] in out:
            continue
        files = []
        for lic in e["license_files"]:
            if licenses is not None:
                p = licenses / e["wheel"] / lic["path"].rsplit("/", 1)[-1]
            else:
                assert roots is not None
                p = roots[e["source"]["root"]] / lic["path"]
            data = p.read_bytes() if p.is_file() else b""
            if _sha256(data) != lic["sha256"]:
                bad.append(f"{p}: not the set's {lic['sha256']}")
            files.append(data)
        out[e["wheel"]] = files
    return out, bad


def _conditions_in_wheel_text(
    audit: dict[str, Any], o: dict[str, Any]
) -> list[str] | None:
    """The object's conditions, searched in its wheel's text; an object whose
    wheel carries no text for it has them checked in the supplied text."""
    if not o["conditions"] or "absent" in o["bundled"]:
        return None
    return audit["conditions"].get(o["conditions"])


def check_bundled(audit: dict[str, Any], shipped: dict[str, list[bytes]]) -> list[str]:
    bad = []
    for o in audit["objects"]:
        # This object's own wheel's files, read and searched for this object.
        text = page_text(b"\n".join(shipped[o["wheel"]]))
        conds = _conditions_in_wheel_text(audit, o)
        bad += _claim_problems(f"{o['soname']} (shipped)", o["bundled"], text, conds)
    return bad


def supplied_bytes(
    audit: dict[str, Any], licenses: Path | None = None
) -> dict[str, bytes]:
    """package path -> bytes of each supplied text: the repo file, or with
    `licenses` (a tree's licenses/) the tree's file."""
    out = {}
    for t in audit.get("supplied_texts", []):
        p = (
            licenses.parent / t["file"]
            if licenses is not None
            else REPO / t["repo_file"]
        )
        out[t["file"]] = p.read_bytes() if p.is_file() else b""
    return out


def check_supplied(
    audit: dict[str, Any], texts: dict[str, bytes], where: str
) -> list[str]:
    """The supplied files carry the recorded bytes, and each object's claims
    about them hold (an object without wheel text also has its conditions
    checked here, across its supplied texts)."""
    bad = []
    for t in audit.get("supplied_texts", []):
        if _sha256(texts.get(t["file"], b"")) != t["sha256"]:
            bad.append(f"{where}: {t['file']} is not sha256 {t['sha256']}")
    for o in audit["objects"]:
        for c in o.get("supplied", []):
            text = page_text(texts.get(c["file"], b""))
            bad += _claim_problems(f"{o['soname']} ({c['file']})", c, text)
        if o.get("supplied") and "absent" in o["bundled"] and o["conditions"]:
            text = page_text(
                b"\n".join(texts.get(c["file"], b"") for c in o["supplied"])
            )
            bad += _claim_problems(
                f"{o['soname']} (supplied)",
                {},
                text,
                audit["conditions"][o["conditions"]],
            )
    return bad


def check_downstream_basis(
    audit: dict[str, Any], shipped: dict[str, list[bytes]]
) -> list[str]:
    """Each quote a downstream clause rests on appears in the wheel licence
    text of every object it is cited for (read per object, never shared)."""
    bad = []
    wheel = {o["soname"]: o["wheel"] for o in audit["objects"]}
    for c in audit["downstream_terms"]["clauses"]:
        for b in c["basis"]:
            for so in clause_objects(audit, b):
                text = page_text(b"\n".join(shipped[wheel[so]]))
                if normalize(b["quote"]) not in text:
                    bad.append(
                        f"downstream {c['id']} ({so}): quote not found: {b['quote'][:80]!r}"
                    )
    return bad


def check_derived_texts(
    audit: dict[str, Any], sources_dir: Path, texts: dict[str, bytes]
) -> list[str]:
    """A supplied official text is the article text of its snapshot, re-derived."""
    bad = []
    for t in audit.get("supplied_texts", []):
        src = t["from"].get("source")
        if src is None:
            continue
        p = sources_dir / audit["sources"][src]["snapshot"]
        try:
            want = terms_text(p.read_bytes()).encode()
        except (OSError, ValueError) as exc:
            bad.append(f"{t['file']}: cannot derive from {p}: {exc}")
            continue
        if texts.get(t["file"]) != want:
            bad.append(f"{t['file']}: not the article text of {src}")
    return bad


def check_corresponding(
    audit: dict[str, Any],
    so: str,
    shipped_object: bytes,
    rpm: bytes,
    srpm: bytes,
    texts: dict[str, bytes],
) -> list[str]:
    """The shipped object against the binary package's file, section by
    section, and the source package against the record and the supplied
    texts taken from it. Comparison, not a rebuild."""
    cs = audit["corresponding_source"][so]
    b, s, corr = cs["binary_package"], cs["source_package"], cs["correspondence"]
    bad = []
    if _sha256(rpm) != b["sha256"]:
        bad.append(f"{so}: binary package is not sha256 {b['sha256']}")
    if _sha256(srpm) != s["sha256"] or len(srpm) != s["bytes"]:
        bad.append(
            f"{so}: source package is not sha256 {s['sha256']} / {s['bytes']} bytes"
        )
    if bad:
        return bad
    tags, _ = rpm_header(rpm)
    if tags.get("SOURCERPM") != s["name"]:
        bad.append(
            f"{so}: binary package SOURCERPM {tags.get('SOURCERPM')!r} != {s['name']}"
        )
    member = rpm_members(rpm).get(b["member"], b"")
    if _sha256(member) != b["member_sha256"]:
        bad.append(f"{so}: {b['member']} is not sha256 {b['member_sha256']}")
        return bad
    mine, theirs = elf_sections(shipped_object), elf_sections(member)
    names = set(mine) | set(theirs)
    expected = (
        set(corr["identical_sections"])
        | set(corr["moved_sections"])
        | set(corr["rewritten_sections"])
    )
    extra = sorted(n for n in names - expected if n and n != ".shstrtab")
    if extra:
        bad.append(f"{so}: sections not accounted for: {extra}")
    for n in [*corr["identical_sections"], *corr["moved_sections"]]:
        x, y = mine.get(n), theirs.get(n)
        if x is None or y is None or x["sha256"] != y["sha256"]:
            bad.append(f"{so}: section {n} differs")
        elif n in corr["identical_sections"] and x["addr"] != y["addr"]:
            bad.append(f"{so}: section {n} moved")
    bid = mine.get(".note.gnu.build-id", {}).get("body", b"")
    if bid[-20:].hex() != corr["build_id"]:
        bad.append(f"{so}: build ID {bid[-20:].hex()} != {corr['build_id']}")
    (syms_a, dyn_a), (syms_b, dyn_b) = elf_dynamic(shipped_object), elf_dynamic(member)
    if syms_a != syms_b:
        bad.append(f"{so}: .dynsym differs beyond section indices")
    relocated = {"5", "10", str(0x6FFFFEF5)}  # DT_STRTAB, DT_STRSZ, DT_GNU_HASH
    keep_a = [e for e in dyn_a if e[0] not in relocated]
    keep_b = [e for e in dyn_b if e[0] not in relocated]
    if sorted(keep_a, key=str) != sorted(
        [*keep_b, ("RPATH", corr["added_rpath"])], key=str
    ):
        bad.append(f"{so}: .dynamic differs beyond the added DT_RPATH")
    srpm_files = rpm_members(srpm)
    wanted: dict[str, list[str]] = {}
    for t in audit["supplied_texts"]:
        f = t["from"]
        if f.get("corresponding_source") == so:
            tarball, name = f["member"].split(":", 1)
            wanted.setdefault(tarball, []).append(name)
            if tarball not in srpm_files and "/" + tarball not in srpm_files:
                bad.append(f"{so}: {tarball} is not in the source package")
    for tarball, members in wanted.items():
        tb = srpm_files.get(tarball) or srpm_files.get("/" + tarball, b"")
        got = tar_xz_members(tb, set(members)) if tb else {}
        for t in audit["supplied_texts"]:
            f = t["from"]
            if f.get("corresponding_source") == so and f["member"].startswith(
                tarball + ":"
            ):
                name = f["member"].split(":", 1)[1]
                if got.get(name) != texts.get(t["file"]):
                    bad.append(f"{so}: {t['file']} is not {f['member']}")
    return bad


def _switcher(data: bytes) -> str | None:
    m = re.search(rb"theme_switcher_version_match = '([^']*)'", data)
    return m.group(1).decode() if m else None


def _archive(url: str) -> str | None:
    m = re.search(r"/archive/(\d[\d.]*)/", url)
    return m.group(1) if m else None


def source_bound(
    o: dict[str, Any],
    source: str,
    audit: dict[str, Any],
    snap: dict[str, bytes],
    release: bool,
) -> tuple[bool | None, str]:
    """Whether `source` is the terms (or release record) of this object's
    version: True, False (a mismatch), or None (unversioned: accepted only as
    the object's own release source, with a risk recorded; reported).

    * a documentation version switcher: the wheel version starts with it;
    * the release notes (the object's release): the wheel version is in the
      object's own component row (RELNOTES_COMPONENT), not anywhere on the page;
    * an archive URL ``/archive/<x>/``: the object's release source is an
      archive page of the same <x> that matches by the rules above;
    * a tag in the URL ``/v<version>[-n]/``: the wheel version."""
    ver = version_of(o["wheel"])
    url = audit["sources"][source]["url"]
    data = snap[source]
    who = f"{o['soname']}: {'release' if release else 'terms'} {source}"
    sw = _switcher(data)
    if sw is not None:
        ok = ver.startswith(sw.lstrip("v"))
        return ok, f"{who} is version {sw}, the wheel is {ver}"
    if release and ("release-notes" in url or "relnotes" in source):
        label = RELNOTES_COMPONENT.get(o["wheel"].rsplit("-", 1)[0])
        if label is None:
            return False, f"{who}: no component row known for {o['wheel']}"
        rows = component_versions(page_text(data), label)
        return rows == [ver], f"{who}: row {label!r} gives {rows}, the wheel is {ver}"
    arch = _archive(url)
    if arch is not None and not release:
        rel_src = o["release"]["source"]
        rel_ok, _ = source_bound(o, rel_src, audit, snap, release=True)
        same = _archive(audit["sources"][rel_src]["url"]) == arch
        return (
            bool(rel_ok) and same,
            f"{who} is archive {arch}; the release {rel_src} is "
            f"{_archive(audit['sources'][rel_src]['url'])} (matched: {rel_ok})",
        )
    if re.search(rf"/v{re.escape(ver)}(-\d+)?/", url):
        return True, f"{who} tag v{ver}"
    if source == o["release"]["source"] and o["risks"]:
        return None, f"{who} not matched to {ver} (unversioned; risk recorded)"
    return False, f"{who} is not tied to version {ver}"


def check_official(
    audit: dict[str, Any], sources_dir: Path, shipped: dict[str, list[bytes]] | None
) -> tuple[list[str], list[str]]:
    """(problems, notes); notes name the release matches that cannot be checked."""
    bad: list[str] = []
    notes: list[str] = []
    snap: dict[str, bytes] = {}
    for k, s in audit["sources"].items():
        p = sources_dir / s["snapshot"]
        data = p.read_bytes() if p.is_file() else b""
        if _sha256(data) != s["sha256"]:
            bad.append(f"source {k}: {p} is not sha256 {s['sha256']}")
        snap[k] = data
    for o in audit["objects"]:
        for c in o["official"]:
            data = snap[c["source"]]
            who = f"{o['soname']} ({c['source']})"
            bad += _claim_problems(who, c, page_text(data))
            for key, same in (
                ("identical_to_bundled", True),
                ("prefix_of_bundled", False),
            ):
                if key not in c:
                    continue
                if shipped is None:
                    bad.append(f"{who}: {key} needs the shipped files")
                    continue
                e_files = shipped[o["wheel"]]
                names = [f for f in e_files if f]
                match = any((f == data) if same else f.startswith(data) for f in names)
                if not match:
                    bad.append(f"{who}: {key} {c[key]} does not hold")
        rel_ok, rel_why = source_bound(
            o, o["release"]["source"], audit, snap, release=True
        )
        if rel_ok is False:
            bad.append(rel_why)
        elif rel_ok is None:
            notes.append(rel_why)
        for c in o["official"]:
            if not grants(c) or c["source"] == o["release"]["source"]:
                continue
            ok, why = source_bound(o, c["source"], audit, snap, release=False)
            if ok is False:
                bad.append(why)
            elif ok is None:
                notes.append(why)
    return bad, notes


def cmd_render(args: argparse.Namespace) -> int:
    audit = _load(args.audit)
    outputs = (
        (args.out, render(_load(args.third_party_set), audit)),
        (args.downstream, render_downstream(audit)),
    )
    if args.check:
        rc = 0
        for out, text in outputs:
            same = out.is_file() and out.read_text() == text
            print(f"{out}: {'is' if same else 'is NOT'} the rendering")
            rc |= not same
        return rc
    for out, text in outputs:
        out.write_text(text)
        print(f"wrote {out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    set_ = _load(args.third_party_set)
    audit = _load(args.audit)
    checks: dict[str, Any] = {}
    checks["coverage"] = {"problems": check_coverage(set_, audit)}

    shipped = None
    if args.licenses or args.purelib:
        roots = None
        if args.licenses is None:
            roots = {"purelib": args.purelib, "nvjpeg_wheel": args.nvjpeg_wheel}
        shipped, files_bad = shipped_files(set_, args.licenses, roots)
        checks["bundled_texts"] = {
            "problems": files_bad + check_bundled(audit, shipped)
        }
    else:
        checks["bundled_texts"] = {"skipped": "no --licenses / --purelib"}

    texts = supplied_bytes(audit)
    supplied_bad = check_supplied(audit, texts, "repo")
    if args.licenses is not None:
        supplied_bad += check_supplied(
            audit, supplied_bytes(audit, args.licenses), "tree"
        )
    checks["supplied_texts"] = {"problems": supplied_bad}
    if shipped is not None:
        checks["downstream_basis"] = {
            "problems": check_downstream_basis(audit, shipped)
        }
    else:
        checks["downstream_basis"] = {"skipped": "no --licenses / --purelib"}

    if args.sources:
        bad, notes = check_official(audit, args.sources, shipped)
        bad += check_derived_texts(audit, args.sources, texts)
        checks["official_terms"] = {"problems": bad, "unmatched_releases": notes}
    else:
        checks["official_terms"] = {"skipped": "no --sources"}

    if args.gomp_rpm and args.srpm:
        cs_bad = []
        for so in audit["corresponding_source"]:
            e = next(e for e in set_["entries"] if e["soname"] == so)
            if args.licenses is not None:
                obj = args.licenses.parent / "lib/vendor" / so
            else:
                root = (
                    args.purelib
                    if e["source"]["root"] == "purelib"
                    else args.nvjpeg_wheel
                )
                obj = (root or Path("/nonexistent")) / e["source"]["path"]
            data = obj.read_bytes() if obj.is_file() else b""
            if _sha256(data) != e["sha256"]:
                cs_bad.append(f"{so}: {obj} is not the set's {e['sha256']}")
                continue
            cs_bad += check_corresponding(
                audit,
                so,
                data,
                args.gomp_rpm.read_bytes(),
                args.srpm.read_bytes(),
                texts,
            )
        checks["corresponding_source"] = {"problems": cs_bad}
    else:
        checks["corresponding_source"] = {"skipped": "no --gomp-rpm / --srpm"}

    notice_ok = args.notice.is_file() and args.notice.read_text() == render(set_, audit)
    down_ok = (
        args.downstream.is_file()
        and args.downstream.read_text() == render_downstream(audit)
    )
    checks["notice"] = {
        "problems": ([] if notice_ok else [f"{args.notice} is not the rendering"])
        + ([] if down_ok else [f"{args.downstream} is not the rendering"])
    }
    for c in checks.values():
        if "skipped" not in c:
            c["pass"] = not c["problems"]
    report = {
        "schema": SCHEMA,
        "audit": str(args.audit),
        "audit_sha256": _sha256(args.audit.read_bytes()),
        "distribution": audit["distribution"]["status"],
        "statuses": {
            s: sorted(o["soname"] for o in audit["objects"] if o["status"] == s)
            for s in audit["statuses"]
        },
        "checks": checks,
        "pass": all(c.get("pass", True) for c in checks.values()),
        "complete": not any("skipped" in c for c in checks.values()),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for k, c in checks.items():
        state = "SKIPPED" if "skipped" in c else ("PASS" if c["pass"] else "FAIL")
        print(f"{k}: {state}")
        for p in c.get("problems", [])[:20]:
            print(f"  {p}")
    return 0 if report["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--third-party-set", type=Path, default=THIRD_PARTY_SET)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", type=Path, default=NOTICE)
    r.add_argument("--downstream", type=Path, default=DOWNSTREAM)
    r.add_argument("--check", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--licenses", type=Path, help="a tree's licenses/ directory")
    c.add_argument(
        "--purelib", type=Path, help="the wheels' site-packages (instead of --licenses)"
    )
    c.add_argument(
        "--nvjpeg-wheel",
        type=Path,
        default=REPO / "build/_deps/saccade_nvjpeg_wheel-src",
    )
    c.add_argument("--sources", type=Path, help="the official-terms snapshots")
    c.add_argument("--notice", type=Path, default=NOTICE)
    c.add_argument("--downstream", type=Path, default=DOWNSTREAM)
    c.add_argument(
        "--gomp-rpm", type=Path, help="the binary package corresponding_source names"
    )
    c.add_argument(
        "--srpm", type=Path, help="the source package corresponding_source names"
    )
    c.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    try:
        return cmd_render(args) if args.cmd == "render" else cmd_check(args)
    except (OSError, KeyError, ValueError) as exc:
        print(f"license_audit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Contract for ADR 021 AP-4: reconstructed provenance, and what may never get it.

Backfill is the step where fabrication is cheapest. The directory is right
there, the schema wants eight fields, and filling the four nobody recorded with
``datetime.now()`` and ``""`` produces a file that validates, looks like every
other manifest, and is false. So the tests here are mostly tests that the tool
*declines*:

* a fact with no named source is **absent** from the payload, never a placeholder;
* a directory holding several runs is never given one manifest, which would be
  the same misattribution ``open_run`` refuses at production time;
* a directory that already attests to itself is not given a second, weaker identity;
* an existing manifest is never replaced.

And one test that it does not decline everything: a single-run directory whose
own metadata file names the commit is eligible, and the manifest it gets says
plainly that it was reconstructed.

The distinguishability of the two modes is the load-bearing schema change. A
production manifest earns trust from the ordering guarantee ``open_run``
enforces; a reconstruction has no such guarantee, and if the file did not say
which it was, a reader would extend the first one's trust to the second.
"""

# scope: system
# function: contract
# lifecycle: active

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.provenance import backfill as bf  # noqa: E402
from scripts.provenance.run_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    ManifestError,
    attach_reconstructed_manifest,
    build_manifest,
    build_reconstructed_manifest,
    open_run,
    provenance_mode_of,
    read_manifest,
    validate_manifest,
)

RUN_META = (
    "git_sha=2bc556f2a2ae19758878fe2b0778634c9a5c2b2b\n"
    "host=TESTHOST\n"
    "date=2026-07-09T10:50:28+08:00\n"
    "gpu=GPU 0: NVIDIA Test\n"
    "preset=mamba_whole_graph detector=SDP double-buffer=yes\n"
)


def _sources(**over):
    payload = {
        "produced_by": "eval",
        "commit": "2bc556f2",
        "backfill_sources": ["results/x/run_meta.txt: git_sha="],
    }
    payload.update(over)
    return payload


# --------------------------------------------------------------------------
# The two modes must be distinguishable, and each must be internally coherent.
# --------------------------------------------------------------------------


def test_a_production_manifest_says_it_was_written_by_the_run():
    assert build_manifest("r", produced_by="eval")["provenance_mode"] == "production"


def test_a_reconstructed_manifest_says_it_was_assembled_afterwards():
    payload = build_reconstructed_manifest("r", **_sources())
    assert payload["provenance_mode"] == "reconstructed"


def test_provenance_mode_is_a_closed_vocabulary():
    payload = build_manifest("r", produced_by="eval")
    payload["provenance_mode"] = "backfilled"
    with pytest.raises(ManifestError, match="provenance_mode must be one of"):
        validate_manifest(payload)


def _v1(**over):
    """A manifest in the shape open_run wrote before v2 existed."""
    payload = build_manifest("r", produced_by="eval")
    payload.pop("provenance_mode")
    payload["schema_version"] = 1
    payload.update(over)
    return payload


def test_a_v1_manifest_still_validates_and_reads_as_legacy_production():
    """v1 has one unambiguous meaning, so reading it recovers a fact.

    The only writer that ever produced a v1 file was ``open_run``, which writes
    before the first result byte; no reconstruction writer existed. Rejecting v1
    would also have opened a transition race: AP-2 has been live on main since
    #330, so a run between the survey and this change writes a valid v1 that
    would turn invalid the moment this landed.
    """
    payload = _v1()
    validate_manifest(payload)
    assert provenance_mode_of(payload) == "production"


def test_a_v1_manifest_can_never_be_read_as_a_reconstruction():
    for field, value in (
        ("provenance_mode", "reconstructed"),
        ("backfill_sources", ["some doc"]),
    ):
        with pytest.raises(ManifestError, match="v1 manifest may not carry"):
            validate_manifest(_v1(**{field: value}))


def test_a_v1_manifest_missing_a_v1_required_field_is_still_fail_closed():
    payload = _v1()
    payload.pop("host")
    with pytest.raises(ManifestError, match="schema_version 1"):
        validate_manifest(payload)


def test_the_writer_only_emits_v2(tmp_path):
    """Compatibility is a read-side concession; nothing new is written at v1."""
    open_run(tmp_path / "r", produced_by="eval")
    assert read_manifest(tmp_path / "r")["schema_version"] == 2
    assert build_reconstructed_manifest("r", **_sources())["schema_version"] == 2


def test_a_reconstruction_may_not_be_written_at_the_legacy_version(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "MOT17-02-SDP.txt").write_text("x")
    payload = build_reconstructed_manifest("run", **_sources())
    payload["schema_version"] = 1
    with pytest.raises(ManifestError, match="v1 exists only to be read"):
        attach_reconstructed_manifest(run, payload)


def test_a_production_manifest_may_not_carry_backfill_sources():
    payload = build_manifest("r", produced_by="eval")
    payload["backfill_sources"] = ["some doc"]
    with pytest.raises(ManifestError, match="backfill_sources is meaningless"):
        validate_manifest(payload)


def test_a_reconstruction_must_name_its_sources():
    with pytest.raises(ManifestError, match="backfill_sources must be"):
        build_reconstructed_manifest("r", **_sources(backfill_sources=[]))


def test_a_reconstruction_without_a_commit_is_not_worth_writing():
    """Null commit is honest for a production manifest and useless for this one.

    A production manifest with a null commit still records host, cmdline and
    start time from direct observation. A reconstruction with a null commit has
    established nothing at all.
    """
    for bad in (None, "", "   "):
        with pytest.raises(ManifestError, match="must name a commit"):
            validate_manifest(
                {
                    "schema_version": 2,
                    "run_id": "r",
                    "provenance_mode": "reconstructed",
                    "produced_by": "eval",
                    "commit": bad,
                    "backfill_sources": ["doc"],
                }
            )


# --------------------------------------------------------------------------
# Unknown facts are absent, never filled in.
# --------------------------------------------------------------------------


def test_unsourced_facts_are_omitted_not_placeheld():
    """The failure this prevents: a schema-valid manifest that states falsehoods.

    ``started_at`` set to now, ``host`` to the machine running the backfill,
    ``cmdline`` to ``[]`` — each would read downstream exactly like a fact.
    """
    payload = build_reconstructed_manifest("r", **_sources())
    for absent in ("started_at", "host", "cmdline", "dirty"):
        assert absent not in payload, f"{absent} was invented"


def test_a_payload_missing_those_fields_still_validates():
    """Absence has to be legal, or the previous test only describes a crash."""
    validate_manifest(build_reconstructed_manifest("r", **_sources()))


def test_sourced_facts_are_kept_with_their_source_named():
    payload = build_reconstructed_manifest(
        "r", host="TESTHOST", started_at="2026-07-09T10:50:28+08:00", **_sources()
    )
    assert payload["host"] == "TESTHOST"
    assert payload["backfill_sources"]


def test_produced_by_is_not_required_on_a_reconstruction():
    payload = _sources()
    payload.pop("produced_by")
    validate_manifest(build_reconstructed_manifest("r", **payload))


def test_produced_by_is_still_a_closed_vocabulary_when_stated():
    payload = build_reconstructed_manifest("r", **_sources())
    payload["produced_by"] = "evaluation"
    with pytest.raises(ManifestError, match="produced_by must be one of"):
        validate_manifest(payload)


def test_production_still_requires_every_directly_observable_field():
    """The reconstructed set is shorter; it is not a way to write a thin production one."""
    payload = build_manifest("r", produced_by="eval")
    payload.pop("host")
    with pytest.raises(ManifestError, match="missing required manifest field"):
        validate_manifest(payload)


# --------------------------------------------------------------------------
# Non-reattribution, and how open_run and attach partition the cases.
# --------------------------------------------------------------------------


def test_attach_refuses_to_replace_an_existing_manifest(tmp_path):
    run = tmp_path / "run"
    open_run(run, produced_by="eval")
    before = (run / MANIFEST_FILENAME).read_bytes()

    with pytest.raises(ManifestError, match="already carries"):
        attach_reconstructed_manifest(
            run, build_reconstructed_manifest("run", **_sources())
        )

    assert (run / MANIFEST_FILENAME).read_bytes() == before
    assert read_manifest(run)["provenance_mode"] == "production"


def test_attach_refuses_a_production_payload(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "MOT17-02-SDP.txt").write_text("x")
    with pytest.raises(ManifestError, match="only a reconstruction"):
        attach_reconstructed_manifest(run, build_manifest("run", produced_by="eval"))
    assert not (run / MANIFEST_FILENAME).exists()


def test_attach_refuses_an_empty_directory(tmp_path):
    """An empty directory is open_run's case, not a reconstruction's."""
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ManifestError, match="no bytes here"):
        attach_reconstructed_manifest(
            run, build_reconstructed_manifest("run", **_sources())
        )


def test_open_run_and_attach_partition_the_cases(tmp_path):
    """The mirror image is deliberate: neither function covers both states."""
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "MOT17-02-SDP.txt").write_text("x")
    with pytest.raises(ManifestError, match="not empty"):
        open_run(occupied, produced_by="eval")
    attach_reconstructed_manifest(
        occupied, build_reconstructed_manifest("occupied", **_sources())
    )
    assert read_manifest(occupied)["provenance_mode"] == "reconstructed"


# --------------------------------------------------------------------------
# Discovery: the authority chain, not a grep of the ledger.
# --------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A workspace with the two authorities and a linked source document."""
    (tmp_path / "docs/research/contracts").mkdir(parents=True)
    (tmp_path / "docs/research/tracker-decision").mkdir(parents=True)
    (tmp_path / "docs/research/evidence_ledger.md").write_text(
        "| 2026-07-09 | `2bc556f2` | preset | SDP | 78.4 | "
        "[source](tracker-decision/ablation.md) |\n",
        encoding="utf-8",
    )
    (tmp_path / "docs/research/contracts/claim_state_registry.md").write_text(
        "state registry, names no asset paths\n", encoding="utf-8"
    )
    (tmp_path / "docs/research/tracker-decision/ablation.md").write_text(
        "raw outputs: `results/ablation_20260709/`\n", encoding="utf-8"
    )
    return tmp_path


def _run_dir(repo, rel, *, meta=RUN_META, files=("MOT17-02-SDP.txt",)):
    path = repo / rel
    path.mkdir(parents=True, exist_ok=True)
    for name in files:
        (path / name).write_text("1 1 0 0 10 10 1 -1 -1 -1\n", encoding="utf-8")
    if meta is not None:
        (path / "run_meta.txt").write_text(meta, encoding="utf-8")
    return path


def _one(repo, path):
    return next(c for c in bf.survey(repo) if c.path == path)


def test_discovery_follows_the_ledger_to_the_source_doc(repo):
    """The ledger maps metrics to a document; only the document names the path."""
    _run_dir(repo, "results/ablation_20260709")
    assert "results/ablation_20260709" in bf.discover(repo)


def test_discovery_does_not_follow_a_second_hop(repo):
    """A document cited by a cited document was never endorsed by an authority."""
    (repo / "docs/research/tracker-decision/deeper.md").write_text(
        "`results/two_hops_out/`\n", encoding="utf-8"
    )
    (repo / "docs/research/tracker-decision/ablation.md").write_text(
        "see [deeper](deeper.md)\n", encoding="utf-8"
    )
    assert "results/two_hops_out" not in bf.discover(repo)


def test_a_missing_authority_is_fail_closed_not_an_empty_candidate_set(repo):
    (repo / "docs/research/evidence_ledger.md").unlink()
    with pytest.raises(bf.BackfillError, match="authority document"):
        bf.discover(repo)


def test_a_path_named_only_outside_the_chain_is_not_a_candidate(repo):
    (repo / "docs/notes.md").write_text("`results/unblessed/`\n", encoding="utf-8")
    _run_dir(repo, "results/unblessed")
    assert "results/unblessed" not in bf.discover(repo)


# --------------------------------------------------------------------------
# Classification. Exactly one class is writable.
# --------------------------------------------------------------------------


def test_a_single_run_with_an_in_directory_record_is_eligible(repo):
    _run_dir(repo, "results/ablation_20260709")
    candidate = _one(repo, "results/ablation_20260709")
    assert candidate.classification == bf.ELIGIBLE
    assert candidate.facts["commit"].startswith("2bc556f2")


def test_produced_by_is_never_inferred_from_the_shape_of_the_record(repo):
    """RUN_META has a preset and a detector. That is not evidence of "eval".

    ``run_meta.txt`` is not a versioned schema and nothing contracts what its
    fields imply, so a rule of that form would write an inference into a field
    that reads downstream like an observed fact — the failure this whole module
    is built to refuse.
    """
    run = _run_dir(repo, "results/ablation_20260709")
    assert "preset=" in RUN_META and "detector=" in RUN_META
    candidate = _one(repo, "results/ablation_20260709")
    assert "produced_by" not in candidate.facts

    bf.backfill(repo, candidate)
    assert "produced_by" not in read_manifest(run)


def test_produced_by_is_written_when_the_record_states_it(repo):
    run = _run_dir(
        repo, "results/ablation_20260709", meta=RUN_META + "produced_by=eval\n"
    )
    bf.backfill(repo, _one(repo, "results/ablation_20260709"))
    assert read_manifest(run)["produced_by"] == "eval"


def test_a_stated_kind_outside_the_vocabulary_is_dropped_not_coerced(repo):
    _run_dir(
        repo, "results/ablation_20260709", meta=RUN_META + "produced_by=evaluation\n"
    )
    assert "produced_by" not in _one(repo, "results/ablation_20260709").facts


def test_backfilling_it_writes_a_manifest_that_says_it_was_reconstructed(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    bf.backfill(repo, _one(repo, "results/ablation_20260709"))

    payload = read_manifest(run)
    assert payload["provenance_mode"] == "reconstructed"
    assert payload["commit"].startswith("2bc556f2")
    assert payload["host"] == "TESTHOST"
    assert any("run_meta.txt" in s for s in payload["backfill_sources"])
    assert "cmdline" not in payload and "dirty" not in payload


def test_a_container_of_runs_is_never_given_one_manifest(repo):
    """The ADR's own example: four arms run twice under one cited path.

    One manifest here would assert that eight runs were one — the same
    misattribution ``open_run`` refuses, arriving through the back door.
    """
    for arm in ("A_both_7seq", "B_cost_only_7seq"):
        _run_dir(repo, f"results/ablation_20260709/{arm}")
    (repo / "results/ablation_20260709/run_meta.txt").write_text(
        RUN_META, encoding="utf-8"
    )

    candidate = _one(repo, "results/ablation_20260709")
    assert candidate.classification == bf.CONTAINER
    assert not (repo / "results/ablation_20260709" / MANIFEST_FILENAME).exists()


def test_a_container_is_refused_even_though_its_facts_are_sourceable(repo):
    """Sourceable facts are not the question; what they would be attached to is.

    Without this the tool would happily write a correct-looking commit and host
    onto a directory that is not one run.
    """
    for arm in ("A_both_7seq", "B_cost_only_7seq"):
        _run_dir(repo, f"results/ablation_20260709/{arm}")
    (repo / "results/ablation_20260709/run_meta.txt").write_text(
        RUN_META, encoding="utf-8"
    )
    candidate = _one(repo, "results/ablation_20260709")
    with pytest.raises(bf.BackfillError, match="only a candidate whose required"):
        bf.backfill(repo, candidate)


def test_a_directory_of_directories_with_nothing_written_at_its_level_is_a_container(
    repo,
):
    for study in ("study_a", "study_b"):
        (repo / "results/ablation_20260709" / study).mkdir(parents=True)
    assert _one(repo, "results/ablation_20260709").classification == bf.CONTAINER


def test_a_single_run_that_happens_to_have_one_subdirectory_stays_eligible(repo):
    """Under-classifying every run with a _per_seq/ as a container would leave
    the whole class permanently unaccountable."""
    _run_dir(repo, "results/ablation_20260709")
    (repo / "results/ablation_20260709/_per_seq").mkdir()
    assert _one(repo, "results/ablation_20260709").classification == bf.ELIGIBLE


def test_no_in_directory_record_means_insufficient_not_a_guess(repo):
    """The citing document *does* state the commit — in prose, in a table cell.

    That binding lives in a reader's head. Accepting it would be the tool
    guessing which of the ledger's commits belongs to which path.
    """
    _run_dir(repo, "results/ablation_20260709", meta=None)
    candidate = _one(repo, "results/ablation_20260709")
    assert candidate.classification == bf.INSUFFICIENT
    assert "run_meta.txt" in candidate.reason


def test_a_record_without_a_commit_is_insufficient(repo):
    _run_dir(
        repo, "results/ablation_20260709", meta="host=TESTHOST\npreset=p detector=SDP\n"
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.INSUFFICIENT


def test_a_record_that_does_not_identify_the_kind_of_run_is_still_eligible(repo):
    """``produced_by`` is optional on a reconstruction, so it can be left unsaid.

    The alternative — requiring it, then deriving it from a rule — is how an
    inference gets laundered into a required field.
    """
    run = _run_dir(
        repo,
        "results/ablation_20260709",
        meta="git_sha=2bc556f2a2ae19758878fe2b0778634c9a5c2b2b\nhost=TESTHOST\n",
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.ELIGIBLE
    bf.backfill(repo, _one(repo, "results/ablation_20260709"))
    payload = read_manifest(run)
    assert "produced_by" not in payload
    assert payload["commit"].startswith("2bc556f2")


def test_a_record_declaring_its_own_authority_blocks_a_second_identity(repo):
    """The H2 shape: a record here already speaks for these bytes."""
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "layer_p.json").write_text(
        json.dumps({"authority": "non_authoritative_pre_seal_engineering"}),
        encoding="utf-8",
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.SELF_ATTESTING


def test_an_ordinary_result_json_is_not_mistaken_for_an_authority_record(repo):
    """Otherwise every run with a _latency_profile.json becomes untouchable."""
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "_latency_profile.json").write_text(
        json.dumps({"mean_ms": 3.1, "frames": 600}), encoding="utf-8"
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.ELIGIBLE


def test_a_seal_covering_this_directorys_own_files_blocks_a_write(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "SHA256SUMS").write_text("abc  MOT17-02-SDP.txt\n", encoding="utf-8")
    assert _one(repo, "results/ablation_20260709").classification == bf.SELF_ATTESTING


def test_a_seal_listing_only_its_own_neighbours_does_not_seal_the_parent(repo):
    """The h2_execution shape, and the reason depth is not identity semantics.

    ``archive/checksums.sha256`` lists four files in ``archive/``. That proves
    the closure of ``archive/`` and says nothing about the root or its siblings,
    so a manifest at the root neither touches it nor is covered by it.
    """
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "archive").mkdir()
    (run / "archive/checksums.sha256").write_text(
        "abc  result.json\ndef  run_spec.json\n", encoding="utf-8"
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.ELIGIBLE


def test_a_seal_that_reaches_back_out_of_its_own_directory_does_seal_the_parent(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "archive").mkdir()
    (run / "archive/checksums.sha256").write_text(
        "abc  ../MOT17-02-SDP.txt\n", encoding="utf-8"
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.SELF_ATTESTING


def test_an_unreadable_seal_is_treated_as_covering(repo):
    """Refusing to write is reversible; writing into an unproven seal is not."""
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "SHA256SUMS").write_text("<binary-or-unknown-format>\n", encoding="utf-8")
    assert bf._checksum_entries(run / "SHA256SUMS") is None
    assert _one(repo, "results/ablation_20260709").classification == bf.SELF_ATTESTING


def test_a_json_seal_pack_is_read_for_its_coverage(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / "SHA256SUMS.json").write_text(
        json.dumps({"files": [{"file": "MOT17-02-SDP.txt", "sha256": "abc"}]}),
        encoding="utf-8",
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.SELF_ATTESTING


def test_a_seal_three_levels_down_belongs_to_a_descendant_not_to_this_directory(repo):
    """Otherwise every parent of one sealed study is misreported as a sealed record."""
    for study in ("study_a", "study_b"):
        (repo / "results/ablation_20260709" / study / "pack").mkdir(parents=True)
    (repo / "results/ablation_20260709/study_a/pack/SHA256SUMS.json").write_text(
        "{}", encoding="utf-8"
    )
    assert _one(repo, "results/ablation_20260709").classification == bf.CONTAINER


# --------------------------------------------------------------------------
# Containment. This is a writer; it never takes a path on trust.
# --------------------------------------------------------------------------


def test_a_traversing_token_is_refused_not_followed(repo):
    (repo / "docs/research/tracker-decision/ablation.md").write_text(
        "`results/x/../../../escaped`\n", encoding="utf-8"
    )
    candidate = next(c for c in bf.survey(repo) if ".." in c.path)
    assert candidate.classification == bf.UNSAFE_PATH


def test_a_traversal_that_lands_back_inside_is_still_refused(repo):
    """Containment is not the only reason to refuse a traversal.

    ``results/x/../ablation_20260709`` resolves somewhere legal, so a check that
    only asked "does it end up inside?" would let it through and then write at a
    path that is not the one the document named.
    """
    (repo / "docs/research/tracker-decision/ablation.md").write_text(
        "`results/x/../ablation_20260709`\n", encoding="utf-8"
    )
    _run_dir(repo, "results/ablation_20260709")
    candidate = next(c for c in bf.survey(repo) if ".." in c.path)
    assert candidate.classification == bf.UNSAFE_PATH


def test_write_re_checks_containment_after_resolution(repo, tmp_path):
    """Discovery classified it; the writer still refuses to take that on trust."""
    _run_dir(repo, "results/ablation_20260709")
    candidate = _one(repo, "results/ablation_20260709")
    outside = bf.Candidate(
        path="../outside_the_repo",
        classification=bf.ELIGIBLE,
        reason="forced",
        facts=dict(candidate.facts),
        sources=candidate.sources,
    )
    with pytest.raises(bf.BackfillError, match="outside the asset roots"):
        bf.backfill(repo, outside)
    assert not (tmp_path.parent / "outside_the_repo").exists()


def test_a_path_outside_the_asset_roots_is_refused(repo):
    _run_dir(repo, "results/ablation_20260709")
    candidate = _one(repo, "results/ablation_20260709")
    with pytest.raises(bf.BackfillError, match="outside the asset roots"):
        bf.backfill(
            repo,
            bf.Candidate(
                path="docs/research",
                classification=bf.ELIGIBLE,
                reason="forced",
                facts=dict(candidate.facts),
                sources=candidate.sources,
            ),
        )


def test_an_already_manifested_directory_is_not_a_backfill_candidate(repo):
    run = _run_dir(repo, "results/ablation_20260709", files=())
    (run / "run_meta.txt").unlink(missing_ok=True)
    for stray in run.iterdir():
        stray.unlink()
    open_run(run, produced_by="eval")
    assert (
        _one(repo, "results/ablation_20260709").classification == bf.ALREADY_MANIFESTED
    )


def test_a_legacy_v1_manifest_is_read_not_crashed_on(repo):
    """The transition-window file the v1 read support exists for.

    Schema compatibility is only half of it: a reader that accepts v1 and a
    consumer that then reaches for payload["provenance_mode"] leaves the same
    directory failing, one frame later and with a worse error.
    """
    run = _run_dir(repo, "results/ablation_20260709")
    (run / MANIFEST_FILENAME).write_text(
        json.dumps(_v1(run_id="ablation_20260709"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidate = _one(repo, "results/ablation_20260709")
    assert candidate.classification == bf.ALREADY_MANIFESTED
    assert "production" in candidate.reason
    assert provenance_mode_of(read_manifest(run)) == "production"


def test_a_legacy_v1_manifest_is_never_overwritten_by_a_backfill(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / MANIFEST_FILENAME).write_text(
        json.dumps(_v1(run_id="ablation_20260709"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = (run / MANIFEST_FILENAME).read_bytes()
    assert bf.main(["--repo-root", str(repo), "--write"]) == 0
    assert (run / MANIFEST_FILENAME).read_bytes() == before


def test_an_invalid_manifest_is_reported_as_broken_not_backfilled(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    assert _one(repo, "results/ablation_20260709").classification == bf.INVALID_MANIFEST


def test_a_cited_file_is_not_promoted_to_its_parent_directory(repo):
    """The document cited a checkpoint. It did not cite the run that made it."""
    (repo / "docs/research/tracker-decision/ablation.md").write_text(
        "weights: `runs/train_x/best.ckpt`\n", encoding="utf-8"
    )
    _run_dir(repo, "runs/train_x", files=("best.ckpt",))
    classes = {c.path: c.classification for c in bf.survey(repo)}
    assert classes["runs/train_x/best.ckpt"] == bf.NOT_A_RUN_DIRECTORY
    assert "runs/train_x" not in classes


def test_a_path_absent_from_this_workspace_is_reported_not_an_error(repo):
    """The asset roots are gitignored; a clean clone legitimately has none of them."""
    assert _one(repo, "results/ablation_20260709").classification == bf.ABSENT


# --------------------------------------------------------------------------
# The tool writes only when told to, and never proposes deleting anything.
# --------------------------------------------------------------------------


def test_the_survey_writes_nothing_without_write(repo, capsys):
    run = _run_dir(repo, "results/ablation_20260709")
    assert bf.main(["--repo-root", str(repo)]) == 0
    assert not (run / MANIFEST_FILENAME).exists()
    assert "dry run" in capsys.readouterr().out


def test_write_only_touches_the_eligible_class(repo, capsys):
    eligible = _run_dir(repo, "results/ablation_20260709")
    (repo / "docs/research/tracker-decision/ablation.md").write_text(
        "`results/ablation_20260709/` and `results/container_20260709/`\n",
        encoding="utf-8",
    )
    for arm in ("A_7seq", "B_7seq"):
        _run_dir(repo, f"results/container_20260709/{arm}")

    assert bf.main(["--repo-root", str(repo), "--write"]) == 0
    assert read_manifest(eligible)["provenance_mode"] == "reconstructed"
    assert not (repo / "results/container_20260709" / MANIFEST_FILENAME).exists()


def test_an_invalid_manifest_fails_the_run_closed(repo):
    run = _run_dir(repo, "results/ablation_20260709")
    (run / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
    assert bf.main(["--repo-root", str(repo)]) == 1


def test_nothing_here_carries_disposal_semantics(repo, capsys):
    """AP-5 is behind owner approval; AP-4 must not pre-empt it with wording."""
    _run_dir(repo, "results/ablation_20260709", meta=None)
    bf.main(["--repo-root", str(repo)])
    text = capsys.readouterr().out.lower()
    for word in (
        "delete",
        "deletable",
        "safe to remove",
        "eligible for removal",
        "days old",
        "stale",
    ):
        assert word not in text, f"disposal vocabulary leaked: {word}"


def test_candidate_carries_no_age_or_size_field():
    assert set(bf.Candidate.__dataclass_fields__) == {
        "path",
        "classification",
        "reason",
        "cited_by",
        "facts",
        "sources",
    }

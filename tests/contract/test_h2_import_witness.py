"""The import witness: proving an execution stayed inside its declared namespace.

W5a declared which repository source bytes are *allowed* to compute a result.
This suite pins the check that they are the bytes that actually did, and the
protection that keeps the resolved authority from being edited while they run.

  * **containment, never equality** — the gate is `observed repo-local code` in
    `declared authority domains`. Demanding that every declared member be
    imported would fail every honest run, since no execution imports the whole
    repository. The direction that matters is that nothing loaded from outside;
  * **observe first, classify second** — `.venv` packages and `build/` objects
    reach the recorder and are classified there, not filtered at its entrance. A
    filter would delete the evidence before anything judged it, and the file
    worth seeing is the one nobody predicted would load;
  * **domains are a set, not a kind** — since `scripts/` became a declared root
    the named execution-semantics tooling under it is also a closure member.
    Admission may not read a precedence order: it asks whether the set is
    non-empty and whether every binding in it agrees on the bytes;
  * **the recorder starts before what it witnesses** — a hook installed in the
    child's `main()` would already have missed that module's top-level imports,
    which are exactly the ones deciding what runs. The bootstrap's own reachable
    set is bounded here rather than assumed;
  * **authority is immutable, working state is not** — a tracker must mutate
    things to track. What it may not mutate is the resolved namespace, and the
    protection is alias-free and recursive, so mutate-then-restore — the edit a
    before/after digest comparison cannot see — has nowhere to happen.

Fixtures are synthesised from the frozen schema (§ 5.3). The two tests that run
the real recorder observe *this* process, which is an execution like any other,
not a replay of producer output.
"""

# scope: tracking, system
# function: contract
# lifecycle: active

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in (REPO_ROOT / "scripts" / "tools", REPO_ROOT / "scripts" / "eval"):
    if _extra.as_posix() not in sys.path:
        sys.path.insert(0, _extra.as_posix())

import h2_import_witness as witness_module  # noqa: E402
import h2_run_spec as run_spec_module  # noqa: E402

WITNESS_CONTRACT = REPO_ROOT / "docs/research/contracts/h2_import_witness_v1.json"


def _fake(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _observation(path: str, *, domains: list[str], sha: str | None = None) -> dict:
    return {
        "authority_domains": sorted(domains),
        "length": 11,
        "loader": "SourceFileLoader",
        "module_names": [path.replace("/", ".")],
        "origin_kind": "source",
        "path": path,
        "sha256": _fake(path) if sha is None else sha,
    }


def _document(*, closure: list[str], named: list[str]) -> dict[str, Any]:
    """A RunSpec-shaped declaration, built from the frozen key names only."""
    closure_members = [
        {"length": 11, "path": path, "sha256": _fake(path)} for path in sorted(closure)
    ]
    named_members = [
        {"length": 11, "path": path, "sha256": _fake(path)} for path in sorted(named)
    ]
    return {
        "execution_semantics_projection": {
            "digest": _fake("projection"),
            "execution_code_closure": {
                "digest": _fake("closure"),
                "members": closure_members,
                "roots": list(run_spec_module.DECLARED_EXECUTION_CODE_ROOTS),
            },
            "members": named_members,
        },
        "execution_semantics_projection_digest": _fake("projection"),
    }


# -- the declaration ----------------------------------------------------


def test_the_declared_roots_cover_the_repository_source_namespace() -> None:
    """`scripts/` is a root, and it is the whole of it.

    The seventeen files the first witnessed execution found unbound all sit under
    `scripts/`. Admitting exactly those, or exactly the two subdirectories they
    occupy, would be choosing roots from an observed call path — the same mistake
    as choosing names from one, which is what W5a was written to undo.
    """
    assert run_spec_module.DECLARED_EXECUTION_CODE_ROOTS == (
        "include/",
        "scripts/",
        "src/",
    )


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/eval/config/core.py",
        "scripts/eval/config/semantic.py",
        "scripts/tools/h2_behavioral_identity.py",
        "scripts/tools/h2_runtime_inputs.py",
        "scripts/tools/resolved_bridge_policy_config.py",
        "scripts/tools/run_h0_phase_a.py",
    ],
)
def test_the_code_the_named_set_missed_is_now_declared(relative: str) -> None:
    """Each of these executes, and before W5b nothing bound its bytes.

    `h2_runtime_inputs` defines the canonicalization every digest in this unit
    rests on; `h2_behavioral_identity` chooses which evaluator stack loads at
    all; `scripts/eval/config` is the policy configuration.
    """
    closure = run_spec_module.execution_code_closure()
    assert relative in {member["path"] for member in closure["members"]}


def test_the_named_tooling_and_the_closure_deliberately_overlap() -> None:
    """The same bytes bound twice, stated rather than left to be noticed.

    Five named execution-semantics members live under `scripts/`, so widening the
    roots made them closure members too. That is double binding, and it is
    intended: it costs a duplicated digest and it means neither authority can be
    edited without the projection moving.
    """
    projection = run_spec_module.execution_semantics_projection()
    named = {member["path"] for member in projection["members"]}
    closure = {
        member["path"] for member in projection["execution_code_closure"]["members"]
    }
    overlap = named & closure
    assert overlap == {
        path
        for path in named
        if path.startswith(run_spec_module.DECLARED_EXECUTION_CODE_ROOTS)
    }
    assert overlap, "widening the roots must have absorbed the tooling members"


# -- classification -----------------------------------------------------


def test_a_file_in_both_authorities_carries_both_domains() -> None:
    """Domains are a set. A classifier returning one kind would have to rank them."""
    document = _document(closure=["scripts/tools/x.py"], named=["scripts/tools/x.py"])
    observed = witness_module.classify(
        [_observation("scripts/tools/x.py", domains=[])],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable={"scripts/tools/x.py"},
    )
    assert observed[0]["authority_domains"] == [
        witness_module.DOMAIN_CLOSURE,
        witness_module.DOMAIN_NAMED,
    ]
    assert not witness_module.containment_failures(observed, document=document)


def test_repository_code_no_authority_binds_fails_closed() -> None:
    """The whole point: a loaded repository file nothing declares stops the run."""
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [_observation("docs/stray.py", domains=[])],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable={"docs/stray.py"},
    )
    assert observed[0]["authority_domains"] == []
    reasons = witness_module.containment_failures(observed, document=document)
    assert reasons == ("unbound repository code loaded: docs/stray.py",)


def test_declared_code_that_loaded_different_bytes_fails_closed() -> None:
    """Membership is not enough; Correction 5 is about the bytes."""
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [_observation("src/a.py", domains=[], sha=_fake("something else"))],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable={"src/a.py"},
    )
    reasons = witness_module.containment_failures(observed, document=document)
    assert reasons == (
        f"{witness_module.DOMAIN_CLOSURE} byte identity differs from the "
        "declaration: src/a.py",
    )


def test_a_declared_member_that_never_loaded_is_not_a_failure() -> None:
    """Containment, not equality. No run imports the whole repository."""
    document = _document(closure=["src/a.py", "src/b.py", "src/c.py"], named=[])
    observed = witness_module.classify(
        [_observation("src/a.py", domains=[], sha=_fake("src/a.py"))],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable={"src/a.py"},
    )
    assert not witness_module.containment_failures(observed, document=document)


def test_an_ignored_dependency_outside_every_root_is_the_environment_axis() -> None:
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [
            _observation(
                ".venv/lib/python3.12/site-packages/yaml/__init__.py", domains=[]
            )
        ],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable=set(),
    )
    assert observed[0]["authority_domains"] == [witness_module.DOMAIN_ENVIRONMENT]


def test_a_native_object_from_an_unbound_build_directory_is_not_excused() -> None:
    """`build/` is ignored, so the environment rule would have waved it through.

    A `.so` under the invocation's bound build directory is carried by the build
    witness. One from any other build directory is carried by nothing, and must
    land unbound rather than be reclassified as an external dependency.
    """
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [
            _observation("build/bound/ext.so", domains=[]),
            _observation("build/other/ext.so", domains=[]),
        ],
        document=document,
        build_dir=REPO_ROOT / "build" / "bound",
        selectable=set(),
    )
    domains = {item["path"]: item["authority_domains"] for item in observed}
    assert domains["build/bound/ext.so"] == [witness_module.DOMAIN_BUILD]
    assert domains["build/other/ext.so"] == []


def test_an_untracked_file_under_a_declared_root_is_not_waved_through() -> None:
    """The case the closure's `--others` selector exists for.

    A file present under a declared root but absent from the closure is either
    mid-edit or misplaced. Either way nothing declares its bytes, and the
    environment rule must not reclassify it as a dependency merely because git
    ignores it.
    """
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [_observation("src/scratch/ad_hoc.py", domains=[])],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable=set(),
    )
    assert observed[0]["authority_domains"] == []


# -- the recorder -------------------------------------------------------


def test_the_recorder_records_dependencies_rather_than_filtering_them() -> None:
    """Third-party code is classified, not dropped at the entrance."""
    recorder = witness_module.ImportRecorder()
    recorder.note_file(
        "yaml",
        (REPO_ROOT / ".venv/lib/python3.12/site-packages/yaml/__init__.py").as_posix(),
        "SourceFileLoader",
    )
    paths = {observation["path"] for observation in recorder.observations()}
    assert ".venv/lib/python3.12/site-packages/yaml/__init__.py" in paths


def test_one_file_under_two_module_names_is_one_observation() -> None:
    """A module name is not a byte identity, so the record is keyed by path."""
    recorder = witness_module.ImportRecorder()
    target = (REPO_ROOT / "scripts/tools/h2_import_witness.py").as_posix()
    recorder.note_file("h2_import_witness", target, "SourceFileLoader")
    recorder.note_file("tools.h2_import_witness", target, "SourceFileLoader")
    observations = recorder.observations()
    assert len(observations) == 1
    assert observations[0]["module_names"] == [
        "h2_import_witness",
        "tools.h2_import_witness",
    ]


def test_a_symlinked_origin_resolves_to_the_file_it_names(tmp_path: Path) -> None:
    """Two paths to one file are one file; recording both would double its identity."""
    link = tmp_path / "aliased.py"
    link.symlink_to(REPO_ROOT / "scripts/tools/h2_import_witness.py")
    recorder = witness_module.ImportRecorder()
    recorder.note_file("aliased", link.as_posix(), "SourceFileLoader")
    assert [observation["path"] for observation in recorder.observations()] == [
        "scripts/tools/h2_import_witness.py"
    ]


def test_code_outside_the_repository_is_not_the_witness_subject() -> None:
    recorder = witness_module.ImportRecorder()
    recorder.note_file("json", json.__file__, "SourceFileLoader")
    assert recorder.observations() == ()


def test_the_recorder_observes_a_real_import_and_digests_what_resolved() -> None:
    """The live path, exercised on this process rather than on a fixture."""
    recorder = witness_module.install()
    try:
        sys.modules.pop("resolved_bridge_policy_config", None)
        import resolved_bridge_policy_config  # noqa: F401
    finally:
        witness_module.uninstall(recorder)
    found = {
        observation["path"]: observation for observation in recorder.observations()
    }
    target = "scripts/tools/resolved_bridge_policy_config.py"
    assert target in found
    payload = (REPO_ROOT / target).read_bytes()
    assert found[target]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert found[target]["origin_kind"] == "source"


def test_only_the_bootstrap_itself_may_precede_the_recorder() -> None:
    """What "installed early enough" means, as a bound rather than a comment."""
    observations = [
        _observation(".venv/lib/python3.12/site-packages/_virtualenv.py", domains=[]),
    ]
    observations[0]["authority_domains"] = [witness_module.DOMAIN_ENVIRONMENT]
    assert not witness_module.bootstrap_failures(
        [
            "scripts/tools/h2_child_bootstrap.py",
            "scripts/tools/h2_import_witness.py",
            ".venv/lib/python3.12/site-packages/_virtualenv.py",
        ],
        observations,
    )
    assert witness_module.bootstrap_failures(
        ["scripts/tools/h2_behavioral_identity.py"], observations
    ) == (
        "repository code loaded before the recorder installed: "
        "scripts/tools/h2_behavioral_identity.py",
    )


def test_the_bootstrap_is_what_the_controller_launches() -> None:
    """The ordering fix is only real if the launch path uses it."""
    import run_h2_measurement as controller

    argv = controller.child_argv(Path("/tmp/run/invocation.json"))
    assert argv[3].endswith("scripts/tools/h2_child_bootstrap.py")
    assert not any(
        argument.endswith("run_h2_measurement_child.py") for argument in argv
    )


# -- the record ---------------------------------------------------------


def test_the_witness_validates_against_its_frozen_contract() -> None:
    """Synthesised from the schema, not captured from the producer (§ 5.3)."""
    import jsonschema

    from h2_runtime_inputs import digest

    observations = sorted(
        [
            _observation("src/a.py", domains=[witness_module.DOMAIN_CLOSURE]),
            _observation(
                "scripts/tools/h2_run_spec.py",
                domains=[witness_module.DOMAIN_CLOSURE, witness_module.DOMAIN_NAMED],
            ),
            _observation(
                ".venv/lib/site-packages/yaml/__init__.py",
                domains=[witness_module.DOMAIN_ENVIRONMENT],
            ),
        ],
        key=lambda observation: observation["path"],
    )
    witness = {
        "algorithm": witness_module.WITNESS_ALGORITHM,
        "authority": witness_module.WITNESS_AUTHORITY,
        "bootstrap": {
            "entry_module": "run_h2_measurement_child",
            "preloaded_repo_local_paths": sorted(witness_module.BOOTSTRAP_SELF_PATHS),
            "recorder_installed_before_entry_import": True,
            "schema": witness_module.BOOTSTRAP_SCHEMA,
        },
        "declared": {
            "execution_code_closure_digest": _fake("closure"),
            "execution_semantics_projection_digest": _fake("projection"),
            "roots": list(run_spec_module.DECLARED_EXECUTION_CODE_ROOTS),
        },
        "digest": digest(observations),
        "observations": observations,
        "schema": witness_module.WITNESS_SCHEMA,
    }
    schema = json.loads(WITNESS_CONTRACT.read_text(encoding="utf-8"))
    jsonschema.validate(witness, schema)
    witness_module.validate_witness(witness)


def test_a_bootstrapped_interpreter_witnesses_its_own_repository_imports() -> None:
    """The ordering claim, exercised in a clean process rather than asserted.

    It cannot be tested in-process: by the time pytest has collected this file
    the repository is already imported, and a recorder installed here would
    report exactly the blind spot the bootstrap exists to remove. So the check
    runs where the real child runs — a fresh isolated interpreter whose first
    repository import is the recorder itself.
    """
    import subprocess

    program = """
import json, sys
from pathlib import Path
REPO = Path(sys.argv[1])
sys.path.insert(0, (REPO / "scripts" / "tools").as_posix())
import h2_import_witness as w
recorder = w.install(entry_module="probe_entry")
import h2_behavioral_identity  # the module the named set had missed
import h2_run_spec
witness = w.build_witness(
    recorder,
    document=h2_run_spec.build_run_spec(),
    build_dir=REPO / "build" / "h2_layer_p",
)
w.validate_witness(witness)
print(json.dumps({
    "preloaded": list(witness["bootstrap"]["preloaded_repo_local_paths"]),
    "observed": {
        item["path"]: item["authority_domains"] for item in witness["observations"]
    },
}))
"""
    completed = subprocess.run(
        [
            (REPO_ROOT / ".venv/bin/python").as_posix(),
            "-I",
            "-B",
            "-c",
            program,
            REPO_ROOT.as_posix(),
        ],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    report = json.loads(completed.stdout.decode("utf-8"))
    observed = report["observed"]

    # The module that decides which evaluator stack loads was invisible to the
    # named content set. It is witnessed now, and bound by the closure.
    assert observed["scripts/tools/h2_behavioral_identity.py"] == [
        witness_module.DOMAIN_CLOSURE
    ]
    assert not [path for path, domains in observed.items() if not domains]

    # Only the recorder itself preceded the recorder.
    external = {
        path
        for path, domains in observed.items()
        if witness_module.DOMAIN_ENVIRONMENT in domains
    }
    assert not (
        set(report["preloaded"]) - witness_module.BOOTSTRAP_SELF_PATHS - external
    )


def test_a_witness_whose_digest_does_not_cover_its_observations_is_refused() -> None:
    document = _document(closure=["src/a.py"], named=[])
    observed = witness_module.classify(
        [_observation("src/a.py", domains=[], sha=_fake("src/a.py"))],
        document=document,
        build_dir=REPO_ROOT / "build" / "absent",
        selectable={"src/a.py"},
    )
    witness = {
        "algorithm": witness_module.WITNESS_ALGORITHM,
        "authority": witness_module.WITNESS_AUTHORITY,
        "bootstrap": {
            "entry_module": "x",
            "preloaded_repo_local_paths": [],
            "recorder_installed_before_entry_import": True,
            "schema": witness_module.BOOTSTRAP_SCHEMA,
        },
        "declared": {},
        "digest": _fake("wrong"),
        "observations": [dict(item) for item in observed],
        "schema": witness_module.WITNESS_SCHEMA,
    }
    with pytest.raises(witness_module.WitnessError, match="digest mismatch"):
        witness_module.validate_witness(witness)


def test_a_witness_carrying_unbound_code_cannot_be_validated() -> None:
    """The archive-side reader reaches the same verdict as the producer."""
    witness = {
        "algorithm": witness_module.WITNESS_ALGORITHM,
        "authority": witness_module.WITNESS_AUTHORITY,
        "bootstrap": {
            "entry_module": "x",
            "preloaded_repo_local_paths": [],
            "recorder_installed_before_entry_import": True,
            "schema": witness_module.BOOTSTRAP_SCHEMA,
        },
        "declared": {},
        "observations": [_observation("docs/stray.py", domains=[])],
        "schema": witness_module.WITNESS_SCHEMA,
    }
    from h2_runtime_inputs import digest

    witness["digest"] = digest(witness["observations"])
    with pytest.raises(witness_module.WitnessError, match="unbound repository code"):
        witness_module.validate_witness(witness)


# -- the authority namespace -------------------------------------------


@pytest.fixture()
def frozen() -> Any:
    document = run_spec_module.build_run_spec()
    return run_spec_module.frozen_authority_namespace(document)


def test_a_top_level_key_cannot_be_reassigned(frozen: Any) -> None:
    with pytest.raises(TypeError):
        frozen["sequences"] = "tampered"


def test_a_nested_mapping_cannot_be_mutated() -> None:
    value = run_spec_module._freeze({"a": {"b": {"c": 1}}}, path="root")
    with pytest.raises(TypeError):
        value["a"]["b"]["c"] = 2


def test_a_nested_sequence_cannot_be_mutated() -> None:
    value = run_spec_module._freeze({"a": [1, [2, 3]]}, path="root")
    with pytest.raises(TypeError):
        value["a"][1][0] = 99
    with pytest.raises(AttributeError):
        value["a"].append(4)


def test_a_retained_alias_to_the_source_cannot_reach_the_authority() -> None:
    """Why the freeze rebuilds instead of wrapping.

    A proxy placed over a mapping the caller still holds protects nothing: the
    caller mutates the mapping it kept and the proxy reports the new value.
    """
    source = {"nested": {"key": "original"}}
    value = run_spec_module._freeze(source, path="root")
    source["nested"]["key"] = "mutated"
    source["added"] = True
    assert value["nested"]["key"] == "original"
    assert "added" not in value


def test_a_consumer_object_is_refused_rather_than_pretended_frozen() -> None:
    """Freezing another object's attributes would leave its internals writable."""

    class Consumer:
        def __init__(self) -> None:
            self.state = []

    with pytest.raises(run_spec_module.RunSpecError, match="unfreezable"):
        run_spec_module._freeze({"consumer": Consumer()}, path="root")


def test_the_working_copy_is_mutable_and_cannot_reach_back(frozen: Any) -> None:
    """A tracker must mutate things to track; it may not mutate these."""
    working = run_spec_module.working_namespace(frozen)
    working["sequences"] = "changed"
    working["_scratch"] = {"depth": [1, 2]}
    working["_scratch"]["depth"].append(3)
    assert frozen["sequences"] != "changed"
    assert "_scratch" not in frozen


def test_mutate_then_restore_has_nowhere_to_happen(frozen: Any) -> None:
    """The edit a before/after digest comparison cannot see.

    This is why the protection is structural. A run that changed a value, used
    it, and put it back would leave every recomputed digest identical.
    """
    original = frozen["sequences"]
    with pytest.raises(TypeError):
        frozen["sequences"] = "temporarily different"
    assert frozen["sequences"] == original


def test_the_working_copy_shares_no_container_with_the_authority(frozen: Any) -> None:
    working = run_spec_module.working_namespace(frozen)
    shared = [
        key
        for key, value in working.items()
        if isinstance(value, (dict, list)) and value is frozen[key]
    ]
    assert shared == []

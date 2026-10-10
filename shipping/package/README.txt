Saccade native tracker: shipping package
=========================================

This tree runs saccade_track, the native multi-object tracker (detector,
tracker and MOT output in one process). It needs no Python, compiler or CUDA
toolkit. Below, <name> is the package name, the "package" field of
MANIFEST.json (for example saccade-0.1.0-linux-x86_64-cu13.0-trt10.16-sm120-glibc2.39).
<prefix> is the directory the package was installed into.


Requirements
------------

GPU     NVIDIA compute capability 12.0 (sm_120) only. The operator library and
        the TensorRT engine carry sm_120 code only. No other GPU has run this
        package.
Driver  An NVIDIA driver with CUDA 13.0 Update 2 support. NVIDIA's release
        notes list >= 580.95.05 for Linux x86_64. This package was verified
        only under WSL2 with Windows driver 616.92 (Linux user-mode driver
        615.71.09). Native Linux hosts have not been verified.
OS      Linux x86_64, glibc >= 2.39 (baseline Ubuntu 24.04), with the system
        loader /lib64/ld-linux-x86-64.so.2. The base system provides
        libstdc++, libgcc_s and zlib. Everything else ships in lib/vendor/.
Disk    About 3.7 GiB installed. The installer also needs room for a staging
        copy next to the target.


Verify, then install
--------------------

A release is three files: <name>.tar.gz, <name>.install.sh and <name>.sha256.
A signed release has a fourth, <name>.sha256.minisig. Packages are local-only
for now and need not be signed.

What each check shows:

  sha256 digest and MANIFEST.json   the files are the ones the digest names,
                                    complete and unmodified (integrity).
  minisign signature                the digest was signed by the holder of
                                    the key you verify it with (publisher
                                    authentication).

A digest is not a signature: whoever can replace the tarball can replace
<name>.sha256 too. An unsigned package is not authenticated, however its
checks turn out; do not describe it as signed or verified.

1. Signed release only. Get the release public key from the Saccade
   repository (shipping/package/minisign.pub), not from where you downloaded
   the package, and check the signature of the digest file:

     minisign -Vm <name>.sha256 -p minisign.pub

   minisign prints the trusted comment:
     package=<name> commit=<source commit> manifest_sha256=<sha256 of MANIFEST.json>
   If it fails, stop: do not install. The installer cannot verify itself;
   this step is what covers it.

2. Check the digest (signed or not):

     sha256sum -c <name>.sha256

3. Install. TARGET must not exist, and its parent directory must exist. The
   path must not contain ':' or ';'.

     sh <name>.install.sh <name>.tar.gz TARGET

   The installer checks the digest, extracts to a staging directory next to
   TARGET, and checks every file against MANIFEST.json. It then renames the
   staging directory to TARGET in one step. On any failure TARGET is not
   created. It does the same for a signed and an unsigned package: it never
   reads the signature.

4. Later checks of an installed tree:

     sh <name>.install.sh --verify TARGET
     sha256sum TARGET/MANIFEST.json

   --verify checks the tree against the MANIFEST.json inside it. For a signed
   release, the sha256 of that file must equal manifest_sha256 in the signed
   trusted comment; otherwise the MANIFEST itself is not vouched for. An
   unsigned package has no such reference.


Run
---

  <prefix>/bin/saccade_track --config JSON --lineage JSON [--attestation JSON]
      [--model-root DIR] --out DIR [--report JSON] [--trace DIR] SEQUENCE_DIR...

With the model root this package carries:

  M=<prefix>/share/saccade
  <prefix>/bin/saccade_track \
      --config $M/configs/shipping/mamba_whole_graph.resolved.json \
      --lineage $M/models/yolo/mamba_head_s_v14replica_t3_t1_fp32_torchscript.lineage.json \
      --attestation $M/configs/shipping/mamba_head_realization.attestation.json \
      --model-root $M --out OUT  MOT17/train/MOT17-02-FRCNN ...

Each SEQUENCE_DIR is a MOT-format sequence (img1/, seqinfo.ini). One MOT text
file per sequence is written to OUT. bin/saccade_track is a launcher. It runs
libexec/saccade_track through the system loader with lib/vendor as the
library path and a loader audit that refuses copies of the bundled libraries
found outside this tree. It exits 127 if that check cannot be loaded.


Completion and failed reruns
---------------------------

After accepting the arguments, saccade_track prints a new run_id on its first
stderr line. It holds an exclusive lock on OUT/saccade_track.lock until it
exits. A concurrent second run using the same OUT exits 2 without changing the first
run's files. The lock file remains after the process exits.

OUT/saccade_track.journal.json records that run_id, a state (running, failed
or complete), and each sequence's pending or written state. Written entries
carry the MOT file's sha256. If --report is requested, its format is
saccade.native_track_report/v3; it carries the same run_id and its hash is
recorded in the journal. Exit 0 and a complete journal for this invocation
are required for a complete run. Check the run_id and recorded hashes before
using its outputs; files from an earlier invocation do not prove success.

Once the lock is acquired and a new journal installed, the run removes the
previous --report, the requested sequences' MOT files, and their requested
trace files. Other sequences and unrelated files are left alone. A failed
rerun can therefore remove previously successful outputs. Use a new OUT,
report path and trace directory when you want to preserve an earlier run.

Before initializing CUDA, preflight Gate A checks config, lineage,
attestation when supplied, model file hashes, sequence metadata and frame
listings, and output directories. A refusal exits 2 and attempts to leave a
failed journal. Gate A does not decode frames: a corrupt JPEG can fail later,
after earlier sequences have been written. Complete is written only after
all sequences and the requested report have been published.

A caught runtime error exits 2 and attempts to mark the journal failed.
SIGKILL or an abrupt loader-audit exit can leave it running; treat that run
as incomplete. A pending entry does not establish that its MOT file was
committed. The package does not resume or roll back a whole run. The identity
level in journal and report is currently null; supplied checksums do not
establish a trusted model source or publisher authentication.


Named limits
------------

- sm_120 only; one WSL2 machine and driver verified; no native Linux host.
- TensorRT is the CUDA 12 build (10.16) next to the CUDA 13.0 libraries.
- The output is the native configuration's. It is not the bytes of any
  headline number. Any number quoted for this package must come from this
  configuration.
- The launcher's /bin/sh is not protected from a caller's LD_PRELOAD. The
  loader audit checks paths and names, not bytes.
- The installer's rename is atomic only on file systems that support
  RENAME_NOREPLACE. Only ext4 has been verified.
- A SIGKILL during installation leaves the staging directory, which you must
  remove yourself. TARGET is never created.
- The OUT lock does not protect a report or trace shared with a run using a
  different OUT. Give concurrent runs distinct output paths.
- Run-time flock and rename have been checked on WSL2 ext4 only. Network and
  Windows-mounted file systems, power-loss durability and cleanup of temp
  files left by a killed run have not been verified.


Licenses
--------

Saccade is Apache-2.0 (licenses/saccade/). The libraries in lib/vendor/ are
third-party components under their own terms, not Apache-2.0:

  licenses/<wheel>/     the license files of the wheel each library came from
  licenses/terms/       the version-matched official NVIDIA terms
  licenses/libgomp/     GNU libgomp's license (GPL-3.0 with the GCC Runtime
                        Library Exception) and where to get its source
                        (SOURCE.txt)
  licenses/THIRD_PARTY.md  each object's license files, the terms read for
                        it, the open items and the distribution status

These files are notices. They are not an agreement between you and the
publisher of this package, and nothing in this package grants rights in the
third-party components beyond their own terms.

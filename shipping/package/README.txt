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

A release is four files: <name>.tar.gz, <name>.install.sh, <name>.sha256 and
<name>.sha256.minisig.

1. Get the release public key from the Saccade repository
   (shipping/package/minisign.pub), not from where you downloaded the package.

2. Check the signature of the digest file, then the digest:

     minisign -Vm <name>.sha256 -p minisign.pub
     sha256sum -c <name>.sha256

   minisign prints the trusted comment:
     package=<name> commit=<source commit> manifest_sha256=<sha256 of MANIFEST.json>
   The installer cannot verify itself. Step 2 is what covers it.

3. Install. TARGET must not exist, and its parent directory must exist. The
   path must not contain ':' or ';'.

     sh <name>.install.sh <name>.tar.gz TARGET

   The installer checks the digest, extracts to a staging directory next to
   TARGET, and checks every file against MANIFEST.json. It then renames the
   staging directory to TARGET in one step. On any failure TARGET is not
   created.

4. Later checks of an installed tree:

     sh <name>.install.sh --verify TARGET
     sha256sum TARGET/MANIFEST.json

   --verify checks the tree against the MANIFEST.json inside it. The sha256
   of that file must equal manifest_sha256 in the signed trusted comment.
   Otherwise the MANIFEST itself is not vouched for.


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


Licenses
--------

Saccade is Apache-2.0 (licenses/saccade/). The libraries in lib/vendor/ are
third-party components under their own terms, not Apache-2.0. See
licenses/THIRD_PARTY.md for each object's license files, the terms read for
it, and the distribution status.

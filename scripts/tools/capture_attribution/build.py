"""Build the CUPTI observer and typed decoders from installed CUDA headers."""

# status: diagnostic

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def build(cuda: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    inc = cuda / "include"
    coverage = {}
    creation_coverage = {}
    unparsed = []
    code = []
    for domain, meta, cbids, prefix, stream_type in (
        (
            "RUNTIME",
            "generated_cuda_runtime_api_meta.h",
            "cupti_runtime_cbid.h",
            "CUPTI_RUNTIME_TRACE_CBID_",
            "cudaStream_t",
        ),
        (
            "DRIVER",
            "generated_cuda_meta.h",
            "cupti_driver_cbid.h",
            "CUPTI_DRIVER_TRACE_CBID_",
            "CUstream",
        ),
    ):
        structs = dict(
            (name, body)
            for body, name in re.findall(
                r"typedef struct \w+\s*\{(.*?)\}\s*(\w+)_params;",
                (inc / meta).read_text(),
                re.S,
            )
        )
        ids = re.findall(rf"\b{prefix}(\w+)\s*=", (inc / cbids).read_text())
        code.append(f"if (domain == CUPTI_CB_DOMAIN_{domain}_API) {{ switch (id) {{")
        coverage[domain] = []
        creation_coverage[domain] = []
        for name in ids:
            if not re.search(
                r"Stream(?:Begin|End|IsCapturing|GetCaptureInfo|Create|Destroy|GetFlags|WaitEvent|Synchronize)|^(?:cuda|cu)Event(?:Record|Destroy)",
                name,
            ):
                continue
            if name not in structs:
                unparsed.append(name)
                continue
            body = structs[name]
            stream = re.search(rf"\b{stream_type}\s+(\w+)\s*;", body)
            created = re.search(rf"\b{stream_type}\s*\*\s*(\w+)\s*;", body)
            if not stream and not created and "EventDestroy" not in name:
                raise ValueError(f"No stream field for {name}")
            coverage[domain].append(name)
            if name.startswith(("cudaStreamCreate", "cuStreamCreate")):
                creation_coverage[domain].append(name)
            code.append(
                f"case {prefix}{name}: {{ const auto *p = static_cast<const {name}_params *>(data->functionParams); e.selected = true;"
            )
            if stream:
                code.append(
                    f"e.stream = reinterpret_cast<uintptr_t>(p->{stream[1]}); e.has_stream = true;"
                )
            if created:
                code.append(
                    f"if (ok && p->{created[1]}) {{ e.stream = reinterpret_cast<uintptr_t>(*p->{created[1]}); e.has_stream = true; }}"
                )
            mode = re.search(
                r"\b(?:cudaStreamCaptureMode|CUstreamCaptureMode)\s+(\w+)\s*;", body
            )
            if mode:
                code.append(f"e.mode = static_cast<int>(p->{mode[1]});")
            elif name in ("cuStreamBeginCapture", "cuStreamBeginCapture_ptsz"):
                code.append(
                    "e.mode = 0; // deprecated driver API has implicit global mode"
                )
            if "Create" in name:
                flags = re.search(r"unsigned int\s+(\w*[Ff]lags)\s*;", body)
                code.append(
                    f'if (ok) {{ e.flags = {("p->" + flags[1]) if flags else "0"}; e.flag_source = "create"; }}'
                )
                priority = re.search(r"\bint\s+(priority)\s*;", body)
                if priority:
                    code.append(
                        f"e.priority = p->{priority[1]}; e.has_priority = true;"
                    )
            if "GetFlags" in name:
                flags = re.search(r"unsigned int\s*\*\s*(\w+)\s*;", body)
                if not flags:
                    raise ValueError(name)
                code.append(
                    f'if (ok && p->{flags[1]}) {{ e.flags = *p->{flags[1]}; e.flag_source = "query"; }}'
                )
            status = re.search(
                r"(?:cudaStreamCaptureStatus|CUstreamCaptureStatus)\s*\*\s*(\w+)\s*;",
                body,
            )
            if status:
                code.append(
                    f"if (ok && p->{status[1]}) e.status = static_cast<int>(*p->{status[1]});"
                )
            event = re.search(r"(?:cudaEvent_t|CUevent)\s+(\w+)\s*;", body)
            if event:
                code.append(f"e.event = reinterpret_cast<uintptr_t>(p->{event[1]});")
                event_flags = re.search(r"unsigned int\s+(flags|Flags)\s*;", body)
                code.append(
                    f"e.event_flags = {('p->' + event_flags[1]) if event_flags else '0'};"
                )
            capture_id = re.search(
                r"(?:unsigned long long|cuuint64_t)\s*\*\s*(id_out)\s*;", body
            )
            if capture_id:
                code.append(
                    f"if (ok && e.status == 1 && p->{capture_id[1]}) {{ e.capture_id = *p->{capture_id[1]}; e.has_capture_id = true; }}"
                )
            code.append("break; }")
        code.append("default: break; } }")
    (output / "decode.inc").write_text("\n".join(code) + "\n")
    source = Path(__file__).with_name("observer.cpp")
    control_source = Path(__file__).with_name("control_owner.cpp")
    observer_command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-g",
        "-shared",
        "-fPIC",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{inc}",
        f"-I{output.resolve()}",
        str(source),
        f"-L{cuda / 'lib'}",
        f"-Wl,-rpath,{cuda / 'lib'}",
        "-lcupti",
        "-o",
        str(output / "observer.so"),
    ]
    subprocess.run(observer_command, check=True)
    control_command = [
        "g++",
        "-std=c++17",
        "-O0",
        "-g",
        "-shared",
        "-fPIC",
        "-fno-omit-frame-pointer",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{inc}",
        str(control_source),
        f"-L{cuda / 'lib'}",
        f"-Wl,-rpath,{cuda / 'lib'}",
        "-lcudart",
        f"-L{cuda / 'lib' / 'stubs'}",
        "-lcuda",
        "-o",
        str(output / "control_owner.so"),
    ]
    subprocess.run(control_command, check=True)
    inputs = [source, control_source, Path(__file__), output / "decode.inc"] + [
        inc / name
        for name in (
            "generated_cuda_runtime_api_meta.h",
            "cupti_runtime_cbid.h",
            "generated_cuda_meta.h",
            "cupti_driver_cbid.h",
        )
    ]
    manifest = {
        "schema": "capture_attribution_observer_build_v2",
        "observer_command": observer_command,
        "control_command": control_command,
        "decoded_callbacks": coverage,
        "stream_creation_callbacks": creation_coverage,
        "unparsed_callbacks": unparsed,
        "sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs + [output / "observer.so", output / "control_owner.so"]
        },
    }
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in coverage.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda", type=Path, default=Path("/opt/cuda/targets/x86_64-linux")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.cuda.resolve(), args.output.resolve())

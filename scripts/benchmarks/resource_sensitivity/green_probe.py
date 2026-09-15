"""Probe actual Green Context resources and PyTorch stream-pool escape."""

# status: diagnostic
import argparse
import json
from pathlib import Path


def probe():
    import torch
    from cuda.bindings import driver as d

    def call(name, *args):
        value = getattr(d, name)(*args)
        if int(value[0]):
            raise RuntimeError(f"{name}: {value[0]}")
        return value[1] if len(value) == 2 else value[1:]

    call("cuInit", 0)
    torch.cuda.init()
    primary = call("cuCtxGetCurrent")
    resource = call(
        "cuDeviceGetDevResource", 0, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    )
    # Populate the pool before changing current context, as existing eval does.
    pooled = torch.cuda.Stream()
    report = {
        "sm_count": resource.sm.smCount,
        "min_partition": resource.sm.minSmPartitionSize,
        "alignment": resource.sm.smCoscheduledAlignment,
        "points": [],
        "full_pipeline_partition_verified": False,
    }
    for count in (8, 16, 24, 32):
        row = {"requested": count}
        green = None
        stream = None
        try:
            split = call("cuDevSmResourceSplitByCount", 1, resource, 0, count)
            groups, actual_groups, _ = split
            if actual_groups != 1:
                raise RuntimeError(f"unexpected group count: {actual_groups}")
            desc = call("cuDevResourceGenerateDesc", groups, 1)
            green = call(
                "cuGreenCtxCreate",
                desc,
                0,
                d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM,
            )
            actual = call(
                "cuGreenCtxGetDevResource",
                green,
                d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
            ctx = call("cuCtxFromGreenCtx", green)
            stream = call(
                "cuGreenCtxStreamCreate",
                green,
                d.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,
            )
            row["actual_sm_count"] = actual.sm.smCount
            row["green_stream_context_verified"] = int(
                call("cuStreamGetCtx", stream)
            ) == int(ctx)
            call("cuCtxSetCurrent", ctx)
            new_stream = torch.cuda.Stream()
            row["new_torch_stream_in_green_context"] = int(
                call("cuStreamGetCtx", new_stream.cuda_stream)
            ) == int(ctx)
            row["existing_torch_stream_in_green_context"] = int(
                call("cuStreamGetCtx", pooled.cuda_stream)
            ) == int(ctx)
            external = torch.cuda.ExternalStream(int(stream), device=0)
            with torch.cuda.stream(external):
                x = torch.ones(32, device="cuda")
                y = x + 1
            external.synchronize()
            row["external_stream_torch_kernel_ok"] = bool((y.cpu() == 2).all())
            del x, y, external
        except Exception as exc:
            row["error"] = str(exc)
        finally:
            call("cuCtxSetCurrent", primary)
            if stream is not None:
                call("cuStreamDestroy", stream)
            if green is not None:
                call("cuGreenCtxDestroy", green)
        report["points"].append(row)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = probe()
    except Exception as exc:
        result = {
            "status": "unavailable",
            "error": str(exc),
            "full_pipeline_partition_verified": False,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")

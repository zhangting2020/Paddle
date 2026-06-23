#!/usr/bin/env python3

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VMM v2 allocator perf benchmark for mapped-free reuse.

This benchmark isolates the non-remap allocator hot path that regressed during
the VMM v2 backing-parts experiments. It intentionally avoids DeepEP, NCCL, and
model kernels:

1. allocate one large VMM tensor to grow the pool,
2. free it to create mapped-free blocks,
3. repeatedly allocate/free smaller tensors from those mapped-free blocks,
4. collect VMM_STEP counters via
   core._vmm_v2_step_stats_snapshot_and_reset(device).

The parent process launches children because Paddle FLAGS must be set before
importing paddle. By default the benchmark compares h=2 and h=16 under:

- legacy:   FLAGS_vmm_v2_legacy_mapped_free_split=1
- optimized:FLAGS_vmm_v2_legacy_mapped_free_split=0
- lazy:     FLAGS_vmm_v2_lazy_block_parts=1

Use this as a local performance regression guard before relying on 4-machine
model throughput. A healthy build should show optimized split clearly slower
than legacy/lazy on h=2 for mapped_free_total_us.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

MI_B = 1024 * 1024
PREFIX = "[VMM_ALLOC_PERF] "


def _set_optional_pythonpath(env: dict[str, str], path: str) -> None:
    if not path:
        return
    build_python = str(Path(path).resolve())
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{build_python}:{old}" if old else build_python


def child_main(args: argparse.Namespace) -> int:
    import paddle

    paddle.set_device(f"gpu:{args.device}")
    core = paddle.base.core
    snapshot = getattr(core, "_vmm_v2_step_stats_snapshot_and_reset", None)
    tensor_info = getattr(core, "vmm_tensor_info", None)
    if snapshot is None:
        raise RuntimeError("VMM v2 step stats API is not available")

    def snap() -> dict:
        out = dict(snapshot(args.device))
        out["available"] = True
        return out

    def emit(
        name: str, stats: dict | None = None, extra: dict | None = None
    ) -> None:
        payload = {"name": name}
        if stats:
            payload.update(stats)
        if extra:
            payload.update(extra)
        print(PREFIX + json.dumps(payload, sort_keys=True), flush=True)

    emit(
        "config",
        extra={
            "label": args.label,
            "device": args.device,
            "handle_mb": args.handle_mb,
            "split_mode": args.split_mode,
            "pool_mb": args.pool_mb,
            "pattern": args.pattern,
            "alloc_mb": args.alloc_mb,
            "steps": args.steps,
            "allocs_per_step": args.allocs_per_step,
            "hold_every": args.hold_every,
        },
    )

    snap()
    base = paddle.empty([args.pool_mb * MI_B], dtype="uint8")
    paddle.device.synchronize()
    base_info = {}
    if tensor_info is not None:
        try:
            base_info = dict(tensor_info(base))
            base_info.pop("parts", None)
        except Exception as exc:  # tensor-info is diagnostic only here
            base_info = {"error": repr(exc)}
    emit("base_alloc", snap(), {"base_info": base_info})

    del base
    gc.collect()
    paddle.device.synchronize()
    emit("base_free", snap())

    held = []
    for step in range(args.steps):
        tensors = []
        sizes = []
        for i in range(args.allocs_per_step):
            if args.pattern == "fixed":
                size_mb = args.alloc_mb
            else:
                # Deterministic 4..64 MiB pattern. It avoids Python RNG cost in
                # the measured path and gives stable mapped-free fragmentation.
                size_mb = 4 + 4 * (
                    ((step * args.allocs_per_step + i) * 17 + 3) % 16
                )
            sizes.append(size_mb)
            tensors.append(paddle.empty([size_mb * MI_B], dtype="uint8"))

        paddle.device.synchronize()
        emit(
            "step_alloc",
            snap(),
            {
                "step": step,
                "alloc_count_step": len(tensors),
                "alloc_mb_total": sum(sizes),
                "alloc_mb_min": min(sizes),
                "alloc_mb_max": max(sizes),
            },
        )

        if args.hold_every > 0:
            kept = []
            freed = []
            for i, tensor in enumerate(tensors):
                global_i = step * args.allocs_per_step + i
                if global_i % args.hold_every == 0:
                    kept.append(tensor)
                else:
                    freed.append(tensor)
            held.extend(kept)
            tensors = freed

        del tensors
        gc.collect()
        paddle.device.synchronize()
        emit("step_free", snap(), {"step": step, "held_count": len(held)})

    del held
    gc.collect()
    paddle.device.synchronize()
    emit("final_free", snap())
    return 0


def run_child(
    args: argparse.Namespace, handle_mb: int, split_mode: str, pattern: str
) -> tuple[int, str]:
    env = os.environ.copy()
    _set_optional_pythonpath(env, args.paddle_build_python)
    env.update(
        {
            "FLAGS_allocator_strategy": "auto_growth",
            "FLAGS_use_vmm_auto_growth_best_fit_allocator_v2": "1",
            "FLAGS_vmm_v2_step_stats": "1",
            "FLAGS_vmm_v2_remap_on_oom": "0",
            "FLAGS_vmm_v2_small_pool_handle_size_in_mb": str(handle_mb),
            "FLAGS_vmm_v2_large_pool_handle_size_in_mb": str(handle_mb),
            "FLAGS_vmm_v2_legacy_mapped_free_split": "1",
            "FLAGS_vmm_v2_lazy_block_parts": "0",
            "GLOG_v": env.get("GLOG_v", "0"),
        }
    )
    if split_mode == "optimized":
        env["FLAGS_vmm_v2_legacy_mapped_free_split"] = "0"
    elif split_mode == "lazy":
        env["FLAGS_vmm_v2_lazy_block_parts"] = "1"
    elif split_mode != "legacy":
        raise ValueError(f"unknown split mode: {split_mode}")

    label = f"h{handle_mb}_{split_mode}_{pattern}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--device",
        str(args.device),
        "--label",
        label,
        "--handle-mb",
        str(handle_mb),
        "--split-mode",
        split_mode,
        "--pattern",
        pattern,
        "--pool-mb",
        str(args.pool_mb),
        "--alloc-mb",
        str(args.alloc_mb),
        "--steps",
        str(args.steps),
        "--allocs-per-step",
        str(args.allocs_per_step),
        "--hold-every",
        str(args.hold_every),
    ]
    proc = subprocess.run(
        cmd, env=env, text=True, capture_output=True, timeout=args.timeout
    )
    return proc.returncode, proc.stdout + proc.stderr


def parse_records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            records.append(json.loads(line[len(PREFIX) :]))
    return records


def sum_field(rows: list[dict], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def max_field(rows: list[dict], key: str) -> int:
    return int(max((int(row.get(key, 0)) for row in rows), default=0))


def fmt_ms(us: int) -> str:
    return f"{us / 1000.0:.3f}ms"


def summarize(records: list[dict]) -> dict:
    alloc_rows = [r for r in records if r.get("name") == "step_alloc"]
    free_rows = [r for r in records if r.get("name") == "step_free"]
    mapped_count = sum_field(alloc_rows, "mapped_free_count")
    source_parts = sum_field(alloc_rows, "mapped_free_source_parts_total")
    alloc_parts = sum_field(alloc_rows, "mapped_free_alloc_parts_total")
    remainder_parts = sum_field(alloc_rows, "mapped_free_remainder_parts_total")
    return {
        "steps": len(alloc_rows),
        "mapped_count": mapped_count,
        "mapped_free_total_us": sum_field(alloc_rows, "mapped_free_total_us"),
        "alloc_total_us": sum_field(alloc_rows, "alloc_total_us"),
        "free_total_us": sum_field(free_rows, "free_total_us"),
        "avg_source_parts": source_parts / mapped_count
        if mapped_count
        else 0.0,
        "avg_alloc_parts": alloc_parts / mapped_count if mapped_count else 0.0,
        "avg_remainder_parts": remainder_parts / mapped_count
        if mapped_count
        else 0.0,
        "max_source_parts": max_field(
            alloc_rows, "mapped_free_source_parts_max"
        ),
        "max_blocks": max_field(alloc_rows, "last_block_count"),
        "max_free_blocks": max_field(alloc_rows, "last_free_blocks"),
    }


def parent_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        "# VMM v2 Allocator Perf Benchmark",
        "",
        "| Handle MiB | Split mode | Pattern | Steps | mapped_free cnt | mapped_free time | alloc total | free total | avg src parts | avg alloc parts | avg rem parts | max src parts | max blocks | max free blocks | raw |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    summaries: dict[tuple[int, str, str], dict] = {}
    ok = True

    for handle_mb in args.handles_mb:
        for split_mode in args.split_modes:
            for pattern in args.patterns:
                code, output = run_child(args, handle_mb, split_mode, pattern)
                label = f"h{handle_mb}_{split_mode}_{pattern}"
                raw_path = out_dir / f"{label}.log"
                raw_path.write_text(output)
                records = parse_records(output)
                if code != 0 or not records:
                    ok = False
                    rows.append(
                        f"| {handle_mb} | {split_mode} | {pattern} | failed exit={code} | | | | | | | | | | | {raw_path} |"
                    )
                    continue
                summary = summarize(records)
                summaries[(handle_mb, split_mode, pattern)] = summary
                rows.append(
                    "| {h} | {mode} | {pattern} | {steps} | {cnt} | {mapped} | "
                    "{alloc} | {free} | {src:.1f} | {ap:.1f} | {rp:.1f} | "
                    "{max_src} | {max_blocks} | {max_free} | {raw} |".format(
                        h=handle_mb,
                        mode=split_mode,
                        pattern=pattern,
                        steps=summary["steps"],
                        cnt=summary["mapped_count"],
                        mapped=fmt_ms(summary["mapped_free_total_us"]),
                        alloc=fmt_ms(summary["alloc_total_us"]),
                        free=fmt_ms(summary["free_total_us"]),
                        src=summary["avg_source_parts"],
                        ap=summary["avg_alloc_parts"],
                        rp=summary["avg_remainder_parts"],
                        max_src=summary["max_source_parts"],
                        max_blocks=summary["max_blocks"],
                        max_free=summary["max_free_blocks"],
                        raw=raw_path,
                    )
                )

    rows.extend(["", "## Ratios", ""])
    rows.append("| Pattern | Ratio | Value |")
    rows.append("|---|---|---:|")
    for pattern in args.patterns:
        h2_legacy = summaries.get((2, "legacy", pattern))
        h2_optimized = summaries.get((2, "optimized", pattern))
        h2_lazy = summaries.get((2, "lazy", pattern))
        h16_legacy = summaries.get((16, "legacy", pattern))
        if h2_legacy and h2_optimized and h2_legacy["mapped_free_total_us"] > 0:
            ratio = (
                h2_optimized["mapped_free_total_us"]
                / h2_legacy["mapped_free_total_us"]
            )
            rows.append(
                f"| {pattern} | h2 optimized / h2 legacy mapped_free | {ratio:.2f}x |"
            )
        if h2_lazy and h2_legacy and h2_legacy["mapped_free_total_us"] > 0:
            ratio = (
                h2_lazy["mapped_free_total_us"]
                / h2_legacy["mapped_free_total_us"]
            )
            rows.append(
                f"| {pattern} | h2 lazy / h2 legacy mapped_free | {ratio:.2f}x |"
            )
        if h2_legacy and h16_legacy and h16_legacy["mapped_free_total_us"] > 0:
            ratio = (
                h2_legacy["mapped_free_total_us"]
                / h16_legacy["mapped_free_total_us"]
            )
            rows.append(
                f"| {pattern} | h2 legacy / h16 legacy mapped_free | {ratio:.2f}x |"
            )

    report = "\n".join(rows) + "\n"
    report_path = out_dir / "summary.md"
    report_path.write_text(report)
    print(report)
    print(f"Report written to {report_path}")
    print(f"Raw logs written under {out_dir}")
    return 0 if ok else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--handles-mb", type=int, nargs="+", default=[2, 16])
    parser.add_argument(
        "--split-modes",
        nargs="+",
        choices=["legacy", "optimized", "lazy"],
        default=["legacy", "optimized", "lazy"],
    )
    parser.add_argument("--handle-mb", type=int, default=16)
    parser.add_argument(
        "--split-mode",
        choices=["legacy", "optimized", "lazy"],
        default="legacy",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["fixed", "random"],
        default=["fixed", "random"],
    )
    parser.add_argument(
        "--pattern", choices=["fixed", "random"], default="fixed"
    )
    parser.add_argument("--pool-mb", type=int, default=4096)
    parser.add_argument("--alloc-mb", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--allocs-per-step", type=int, default=32)
    parser.add_argument("--hold-every", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument(
        "--paddle-build-python",
        default="",
        help="Optional build/python path. Leave empty to use the current Python environment.",
    )
    parser.add_argument("--output-dir", default="vmm_v2_allocator_perf_bench")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

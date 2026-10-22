"""Reject a wrong device allocation before vLLM creates any CUDA context."""
import csv
import io
import json
import os
import subprocess


def main():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    rows = list(csv.reader(io.StringIO(result)))
    visible = {row[0].strip(): int(row[1]) for row in rows}
    allowed = set(os.environ["NIC_ALLOWED_GPUS"].split(","))
    count = int(os.environ["NIC_EXPECTED_GPU_COUNT"])
    threshold = int(os.environ["NIC_MAX_EXISTING_MEMORY_MIB"])
    if len(visible) != count or not set(visible) <= allowed:
        raise SystemExit(
            f"GPU allocation rejected: visible={visible}; permitted={sorted(allowed)}. "
            "Check the DRA claim selects only the configured GPU UUIDs. "
            "Do not override NVIDIA_VISIBLE_DEVICES or bypass the device plugin."
        )
    if any(memory > threshold for memory in visible.values()):
        raise SystemExit(f"Allocated GPU is busy: {visible}; no model was loaded.")
    print(json.dumps({"gpu_guard": "passed", "devices": visible}), flush=True)


if __name__ == "__main__":
    main()

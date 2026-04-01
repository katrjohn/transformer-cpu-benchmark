"""
benchmark.py
============
Generic CPU inference benchmark for any BERT-family model on Hugging Face.

Measures:
  - Parameters, MACs, FLOPs  (via DeepSpeed FLOPs profiler)
  - Mean and p95 latency      (over N forward passes on CPU)

Usage:
    # Any HuggingFace encoder model
    python benchmark.py --model bert-base-uncased
    python benchmark.py --model bert-base-multilingual-cased

    # Custom model with a separate tokenizer
    python benchmark.py --model katrjohn/TinyGreekNewsBERT \\
                        --tokenizer nlpaueb/bert-base-greek-uncased-v1 \\
                        --trust-remote-code

    # Tune the run
    python benchmark.py --model bert-base-uncased --runs 1000 --seq-len 128

Requirements:
    pip install torch transformers deepspeed numpy
"""

import argparse
import contextlib
import logging
import os
import time

# ── Silence noisy libraries before anything is imported ──────────────────────

# 1. Env vars: suppress TF C++ logs and tokenizer fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# 2. Python logging: suppress DeepSpeed/HuggingFace INFO before import
logging.basicConfig(level=logging.WARNING)
for _noisy in ("deepspeed", "transformers", "torch"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect file-descriptor 2 to /dev/null to silence C-level stderr (TF/absl/CUDA)."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


# 3. Import DeepSpeed inside the fd-redirect to eat TF/absl/CUDA C++ errors
import numpy as np
import torch
with _suppress_c_stderr():
    from deepspeed.profiling.flops_profiler import get_model_profile
from transformers import AutoModel, AutoTokenizer

# ── Defaults ──────────────────────────────────────────────────────────────────

SEQ_LEN     = 512
WARM_UP     = 20
RUNS        = 10_000
SAMPLE_TEXT = "The government announced new support measures for workers today."


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, tokenizer_id: str, trust_remote_code: bool = False,
               device: torch.device = torch.device("cpu")):
    print(f"[+] Loading tokenizer : {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    print(f"[+] Loading model     : {model_id}")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = model.to(device).eval()

    return model, tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_forward(model, encoded: dict):
    """Forward pass using only arguments accepted by model.forward()."""
    accepted = set(model.forward.__code__.co_varnames)
    filtered = {k: v for k, v in encoded.items() if k in accepted}
    return model(**filtered)


def resolve_device(requested: str) -> torch.device:
    """Resolve 'cpu', 'cuda', or 'cuda:N' to a torch.device."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def device_label(device: torch.device) -> str:
    """Human-readable device label for the results header."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"GPU ({name})"
    return "CPU"


def profile_flops(model, tokenizer, seq_len: int = SEQ_LEN) -> dict:
    """Return FLOPs, MACs, and parameter count via DeepSpeed (always runs on CPU)."""
    encoded = tokenizer(
        " ".join(["the"] * seq_len),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    encoded = {k: v.to("cpu") for k, v in encoded.items()}
    accepted = set(model.forward.__code__.co_varnames)
    kwargs = {k: v for k, v in encoded.items() if k in accepted}

    # DeepSpeed profiler requires model and inputs on CPU
    original_device = next(model.parameters()).device
    model.to("cpu")

    with torch.no_grad():
        logging.disable(logging.INFO)
        try:
            flops, macs, params = get_model_profile(
                model=model,
                kwargs=kwargs,
                warm_up=10,
                detailed=False,
                print_profile=False,
                as_string=False,
            )
        finally:
            logging.disable(logging.NOTSET)

    # Restore model to original device
    model.to(original_device)

    return {"flops": flops, "macs": macs, "params": params}


# ── Latency benchmark ─────────────────────────────────────────────────────────

def benchmark_latency(
    model,
    tokenizer,
    device: torch.device,
    text: str = SAMPLE_TEXT,
    warm: int = WARM_UP,
    runs: int = RUNS,
) -> dict:
    """Return mean and p95 latency in milliseconds over `runs` forward passes."""
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # GPU timing requires explicit sync; CPU uses perf_counter directly
    use_cuda = device.type == "cuda"

    def _timed_forward():
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _safe_forward(model, encoded)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)   # ms
        else:
            t0 = time.perf_counter()
            _safe_forward(model, encoded)
            return (time.perf_counter() - t0) * 1_000

    with torch.inference_mode():
        for _ in range(warm):
            _safe_forward(model, encoded)

        latencies = [_timed_forward() for _ in range(runs)]

    return {
        "mean_ms": float(np.mean(latencies)),
        "p95_ms":  float(np.percentile(latencies, 95)),
        "runs":    runs,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def print_results(model_id: str, device: torch.device,
                  flops_data: dict, latency_data: dict) -> None:
    label = model_id.split("/")[-1]
    hw    = device_label(device)
    sep   = "=" * 50
    print(f"\n{sep}")
    print(f"  {label}  |  {hw} Benchmark Results")
    print(sep)
    print(f"  Parameters  : {flops_data['params'] / 1e6:>8.1f} M")
    print(f"  MACs        : {flops_data['macs']   / 1e9:>8.2f} GMac")
    print(f"  FLOPs       : {flops_data['flops']  / 1e9:>8.2f} GFLOPs  (2 × MACs)")
    print(sep)
    print(f"  Device      : {hw}")
    print(f"  Runs        : {latency_data['runs']:>8,}")
    print(f"  Mean latency: {latency_data['mean_ms']:>8.2f} ms")
    print(f"  p95  latency: {latency_data['p95_ms']:>8.2f} ms")
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic CPU/GPU inference benchmark for BERT-family models"
    )
    parser.add_argument(
        "--model",     type=str, required=True,
        help="HuggingFace model ID  (e.g. bert-base-uncased)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HuggingFace tokenizer ID (defaults to --model)"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True for custom model architectures"
    )
    parser.add_argument(
        "--device",  type=str, default="cpu",
        help="Device: cpu | cuda | cuda:0  (default: cpu)"
    )
    parser.add_argument(
        "--runs",    type=int, default=RUNS,
        help=f"Number of latency runs (default: {RUNS:,})"
    )
    parser.add_argument(
        "--warm",    type=int, default=WARM_UP,
        help=f"Warm-up passes before timing (default: {WARM_UP})"
    )
    parser.add_argument(
        "--seq-len", type=int, default=SEQ_LEN,
        help=f"Sequence length for FLOPs profiling (default: {SEQ_LEN})"
    )
    parser.add_argument(
        "--text",    type=str, default=SAMPLE_TEXT,
        help="Sample text for latency benchmark"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer_id = args.tokenizer or args.model
    device = resolve_device(args.device)
    print(f"[+] Device            : {device_label(device)}")

    model, tokenizer = load_model(
        model_id=args.model,
        tokenizer_id=tokenizer_id,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )

    print(f"\n[+] Profiling FLOPs (seq_len={args.seq_len}, always on CPU) ...")
    flops_data = profile_flops(model, tokenizer, seq_len=args.seq_len)

    print(f"[+] Benchmarking latency ({args.runs:,} runs, {args.warm} warm-up) ...")
    latency_data = benchmark_latency(
        model, tokenizer,
        device=device,
        text=args.text,
        warm=args.warm,
        runs=args.runs,
    )

    print_results(args.model, device, flops_data, latency_data)


if __name__ == "__main__":
    main()

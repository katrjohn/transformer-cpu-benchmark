# bert-cpu-benchmark

A lightweight, CPU inference benchmark for any **BERT-family encoder model** on Hugging Face.

Measures parameters, MACs, FLOPs, mean latency, and p95 latency — all on CPU, no GPU required.

---

## What it measures

| Metric | Description |
|---|---|
| **Parameters** | Total model parameter count |
| **MACs** | Multiply–accumulate operations per forward pass |
| **FLOPs** | Floating-point operations per forward pass (= 2 × MACs) |
| **Mean latency** | Average CPU inference time across N runs (ms) |
| **p95 latency** | 95th-percentile CPU inference time (ms) |

> FLOPs are **hardware-agnostic** — they measure the computational cost of the model architecture, not the speed of the machine. Latency is hardware-dependent.

---

## Requirements

```bash
pip install torch transformers deepspeed numpy
```

> DeepSpeed is used only for the FLOPs profiler. The latency benchmark runs on standard PyTorch.

---

## Usage

```bash
# Any HuggingFace encoder model
python benchmark.py --model bert-base-uncased
python benchmark.py --model FacebookAI/xlm-roberta-base
python benchmark.py --model microsoft/deberta-v3-base

# Model with a separate tokenizer (e.g. custom distilled models)
python benchmark.py --model katrjohn/TinyGreekNewsBERT \
                    --tokenizer nlpaueb/bert-base-greek-uncased-v1 \
                    --trust-remote-code

# Tune the run
python benchmark.py --model bert-base-uncased --runs 1000 --seq-len 128

# Custom sample text
python benchmark.py --model bert-base-uncased \
                    --text "The government announced new measures today."
```

### All options

```
--model             HuggingFace model ID (required)
--tokenizer         HuggingFace tokenizer ID (defaults to --model)
--trust-remote-code Pass trust_remote_code=True for custom architectures
--runs              Number of latency runs        (default: 10 000)
--warm              Warm-up passes before timing  (default: 20)
--seq-len           Sequence length for FLOPs     (default: 512)
--text              Sample text for latency test
```

---

## Sample output

```
[+] Loading tokenizer : bert-base-uncased
[+] Loading model     : bert-base-uncased

[+] Profiling FLOPs (seq_len=512) ...
[+] Benchmarking latency (10,000 runs, 20 warm-up) ...

==================================================
  bert-base-uncased  |  CPU Benchmark Results
==================================================
  Parameters  :    109.5 M
  MACs        :    11.17 GMac
  FLOPs       :    22.35 GFLOPs  (2 × MACs)
==================================================
  Runs        :   10,000
  Mean latency:    52.40 ms
  p95  latency:    54.80 ms
==================================================
```

---

## Benchmark results

Results measured on CPU at `seq_len=512`, 10 000 runs, using [`katrjohn/TinyGreekNewsBERT`](https://huggingface.co/katrjohn/TinyGreekNewsBERT) as the primary model under evaluation.

| Model | Params | MACs | FLOPs | Mean latency | p95 latency |
|---|---|---|---|---|---|
| [`katrjohn/TinyGreekNewsBERT`](https://huggingface.co/katrjohn/TinyGreekNewsBERT) | **14.1 M** | **3.23 GMac** | **6.46 GFLOPs** | **13.59 ms** | **14.50 ms** |
| [`katrjohn/XLMRobertaGreekNews`](https://huggingface.co/katrjohn/XLMRobertaGreekNews) | 278.7 M | 48.33 GMac | 96.71 GFLOPs | 140.93 ms | 148.68 ms |
| [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | 183.8 M | 60.40 GMac | 120.88 GFLOPs | 245.55 ms | 261.96 ms |

> TinyGreekNewsBERT achieves **~20× fewer FLOPs** and **~10× lower CPU latency** than XLM-RoBERTa, while remaining within 5 F1 points on NER. See the [full paper](https://ieeexplore.ieee.org/document/11148234) for the complete model comparison.

---

## Compatible models

Works with any `AutoModel`-loadable **encoder-only** (BERT-family) model from Hugging Face, including:

- `bert-*` (Google)
- `roberta-*`, `xlm-roberta-*` (Meta)
- `deberta-*` (Microsoft)
- `electra-*`, `bigbird-*` (Google)
- `distilbert-*`
- `sentence-transformers/*`
- Custom distilled models with `trust_remote_code`

> **Note:** Decoder models (GPT-2, LLaMA) and encoder-decoder models (T5, BART) are not supported — they require different input handling and forward pass logic.

---

## Methodology

- FLOPs profiled via the [DeepSpeed FLOPs Profiler](https://www.deepspeed.ai/tutorials/flops-profiler/) on a synthetic input at the specified `seq_len`.
- Latency measured with `torch.inference_mode()` after warm-up passes to avoid cold-start bias.
- Only arguments accepted by `model.forward()` are passed, ensuring compatibility across architectures.
- Models are loaded from the Hugging Face Hub. An internet connection is required on first run; subsequent runs use the local cache.

---

## Author

**Ioannis Katranis** — [Hugging Face](https://huggingface.co/katrjohn) · [GitHub](https://github.com/katrjohn)

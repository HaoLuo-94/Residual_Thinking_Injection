# Residual Thinking Injection (RTI)

Residual Thinking Injection (RTI) is a training-free inference-time method for improving LLM reasoning.

During decoding, the model normally converts the full next-token distribution into one discrete token and discards the remaining probability information. RTI reuses this discarded signal by constructing a residual direction:

$$
\Delta = \mathbf{e}*{soft} - \mathbf{e}*{hard}
$$

where $\mathbf{e}{\mathrm{soft}}$ is the Top-K soft embedding from logits, and $\mathbf{e}{\mathrm{hard}}$ is the embedding of the emitted token.

RTI keeps the generated token sequence unchanged. It only injects $\alpha\Delta$ into the layer-0 residual stream before the MLP.

---

## Key Features

* Training-free: no fine-tuning or parameter updates.
* Discrete decoding path unchanged.
* Injects residual direction instead of replacing input embeddings.
* Supports baseline / RTI comparison.
* Supports batch evaluation, multi-dataset evaluation, and pass@k.
* Built on vLLM V1.

---

## Method

At decoding step $t$, the model produces logits $z_t$. RTI takes the Top-K tokens and builds:

$$
\mathbf{e}_{soft,t}
=\sum_{j=1}^{K}
p_{j,t}\mathbf{e}*{i*{j,t}}
$$

The emitted token embedding is:

$$
\mathbf{e}_{hard,t}=E[v_t]
$$

Then RTI computes:

$$
\Delta_t=\mathbf{e}*{soft,t}-\mathbf{e}*{hard,t}
$$

Since $\Delta_t$ is available only after step $t$ finishes, it is injected into the next decoding step:

$$
\tilde{\mathbf{r}}^1_{t+1}
=\mathbf{r}^1_{t+1}
+
\alpha m_{t+1}\Delta_t
$$

where:

* $\alpha$ is the injection strength.
* $m_t$ is the injection phase mask.
* $\mathbf{r}^1$ is the layer-0 post-attention residual stream.

---

## Quick Start

The main launcher is:

```bash
bash examples/run_batch.sh
```

Edit the configurable parameters at the top of `examples/run_batch.sh` before running.

Minimal config:

```bash
MODEL="/path/to/model"
DATA_ROOT="/path/to/data"
DATASET_NAME="math500"
OUTPUT_DIR="./outputs"
```

---

## Run Baseline

```bash
BASELINE=1
bash examples/run_batch.sh
```

Baseline mode runs the native vLLM model without residual injection.

---

## Run RTI

```bash
BASELINE=0
ALPHA=0.5
RESIDUAL_TOP_K=16
INJECT_PHASE="think"
INJECT_LAYER=0

bash examples/run_batch.sh
```

For thinking models such as Qwen3 / Qwen3.5, use:

```bash
INJECT_PHASE="think"
```

For non-thinking models, use:

```bash
INJECT_PHASE="all"
```

---

## Main Parameters

| Parameter        | Description                   |
| ---------------- | ----------------------------- |
| `MODEL`          | Model path                    |
| `DATA`           | One or more jsonl files       |
| `DATA_ROOT`      | Dataset root                  |
| `DATASET_NAME`   | Dataset evaluator             |
| `OUTPUT_DIR`     | Output directory              |
| `BASELINE`       | `1` = baseline, `0` = RTI     |
| `ALPHA`          | Injection strength            |
| `RESIDUAL_TOP_K` | Top-K for building $\Delta$   |
| `INJECT_PHASE`   | `think` or `all`              |
| `INJECT_LAYER`   | `0` for layer-0 injection     |
| `NUM_SAMPLES`    | Number of samples per problem |
| `TEMPERATURE`    | Sampling temperature          |
| `TOP_P`          | Sampling top-p                |
| `TOP_K`          | Sampling top-k                |

Supported datasets:

```text
math500
aime2025
aime2325
minerva
mbpp
humaneval
livecodebench
ifeval
```

---

## Pass@k

For pass@k evaluation, set:

```bash
NUM_SAMPLES=8
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
```

Then run:

```bash
bash examples/run_batch.sh
```

---

## Output

Each run creates:

```text
OUTPUT_DIR/batch_<timestamp>/
```

Main files:

```text
launch_args.txt
run.log
batch_output.jsonl
batch_output.summary.json
summary_all.json
```

For multi-dataset runs, each dataset has its own output file, and all results are merged into `summary_all.json`.

---

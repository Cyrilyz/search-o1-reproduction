# Reproducibility Study: Search-o1 (Li et al., 2025)

**CSE 517 — Spring 2025 Reproducibility Project**

This repository contains our reproduction of [Search-o1: Agentic Search-Enhanced Large Reasoning Models](https://arxiv.org/abs/2501.05366) (Li et al., 2025).

We intentionally kept the notebook format to preserve the full debugging history, which is itself relevant to a reproducibility study. Each cell documents not just what we ran, but what failed, how we diagnosed it, and what we changed — providing a transparent record of the reproduction process.

---

## Original Paper's Repository

**Original repo:** [https://github.com/sunnynexus/Search-o1](https://github.com/sunnynexus/Search-o1)

We used the original authors' code as our starting point and made targeted modifications to adapt it to our available infrastructure.

---

## Repository Structure

```
.
├── README.md                                  # This file
├── QwQ_7B_Search_o1.ipynb                      # Main experiment notebook (7B model, all datasets)
└── QwQ_32B_AWQ_Search_o1.ipynb                 # 32B model experiments (GPQA only)
```

### Notebook 1: `QwQ_7B_Search_o1.ipynb` (7B Model — All Datasets)

| Cells     | Section                                  | Description                                                  |
|-----------|------------------------------------------|--------------------------------------------------------------|
| 0–4       | 1. Environment Setup                    | Clone repo, install dependencies (vLLM, transformers, etc.)  |
| 5–8       | 2. Read Original `bing_search.py`        | Understand original API signatures before replacement        |
| 9–10      | 3. Set Brave API Key                     | Configure search credentials                                 |
| 11–14     | 4. Test Brave Search API                 | Verify Brave Search returns usable results                   |
| 15–16     | 5. Create Drop-in Replacement            | Replace `bing_search.py` with Brave Search drop-in           |
| 17–21     | 6. Verify Replacement Works              | Unit-test the new search module                              |
| 22        | 7. Quick Integration Test                | Ensure `run_search_o1.py` imports cleanly                    |
| 23–33     | Prepare GPQA & Model Setup               | Download GPQA, convert format, fix pyext, load tokenizer     |
| 34–48     | GPQA — Direct, Naive RAG, RAgent (7B)    | Run + evaluate all three baseline methods                    |
| 49–50     | GPQA — Search-o1 (7B)                    | Run Search-o1 on GPQA                                        |
| 51–57     | GPQA — Search-o1 Analysis                | Search trigger analysis (0/198 triggered on 7B)              |
| 58–59     | Prepare NQ & HotpotQA Datasets           | Download and convert both datasets                           |
| 60–65     | NQ — All 4 Methods (7B)                  | Direct, Naive RAG, RAgent, Search-o1                         |
| 66–74     | HotpotQA — All 4 Methods (7B)            | Direct, Naive RAG, RAgent, Search-o1                         |
| 75–76     | Results Aggregation                      | Collect all `.metrics.json` and display summary               |

### Notebook 2: `QwQ_32B_AWQ_Search_o1.ipynb` (32B Model — GPQA)

| Cells     | Section                                  | Description                                                  |
|-----------|------------------------------------------|--------------------------------------------------------------|
| 0–5       | 0. Environment Setup                    | Clone repo, install deps, fix pyext                          |
| 6–7       | 1. Restore Brave Search Replacement      | Write the finalized `bing_search.py` drop-in                 |
| 8–9       | 2. Set Brave API Key                     | Load from Drive or manual input                              |
| 10–13     | 3. Prepare GPQA Diamond Dataset          | Download and convert                                         |
| 14–16     | 4. Quick Model Test (3 questions)        | Verify 32B model loads and generates correctly               |
| 17–26     | 5. Run All Methods (32B on GPQA)         | Search-o1, Naive RAG, RAgent, Direct Reasoning               |
| 27–31     | 6. Analyze Results                       | Search trigger analysis (111/198 triggered)                  |
| 32–35     | 7. Direct Reasoning Baseline             | Run direct gen for backoff comparison                        |
| 36–38     | 8. Backoff Analysis (Optional)           | Apply backoff evaluation                                     |
| 39–40     | 9. Final 7B vs 32B Comparison            | Side-by-side results                                         |

---

## Dependencies

The following packages are required. All are installed in Cell 4 of the notebook.

- **Python** 3.10+
- **vLLM** 0.6.4+ (latest version required for Blackwell GPU compatibility)
- **PyTorch** 2.x with CUDA support
- **transformers** (latest)
- **datasets** (HuggingFace)
- **requests** (for Brave Search API)
- **trafilatura** (installed but ultimately unused — we use snippet-based retrieval)
- **pyext** — replaced with a dummy module (only needed for LiveCodeBench, not our datasets):
  ```bash
  mkdir -p /usr/local/lib/python3.12/dist-packages/pyext
  echo "class RuntimeModule: pass" > /usr/local/lib/python3.12/dist-packages/pyext/__init__.py
  ```

Install all dependencies:
```bash
git clone https://github.com/sunnynexus/Search-o1.git
cd Search-o1
pip install vllm transformers datasets requests trafilatura
```

---

## Data Download Instructions

### GPQA Diamond (198 questions)
GPQA is a gated dataset. Two options:

**Option A — Gated access (requires HuggingFace account approval):**
```bash
huggingface-cli login
# Then in Python:
from datasets import load_dataset
ds = load_dataset("idavidrein/gpqa", "gpqa_diamond", split="train")
```

**Option B — Public mirror:**
```python
from datasets import load_dataset
ds = load_dataset("fingertap/GPQA", split="train")
# Filter to diamond subset
```

### Natural Questions (500 questions)
```python
from datasets import load_dataset
ds = load_dataset("google-research-datasets/natural_questions", split="validation")
# Subsample 500 questions and convert to Search-o1 format
```

### HotpotQA (500 questions)
```python
from datasets import load_dataset
ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
# Subsample 500 questions and convert to Search-o1 format
```

Dataset conversion to Search-o1's expected JSON format is handled in Cells 30–36 (GPQA) and Cell 91 (NQ, HotpotQA) of the notebook.

---

## Preprocessing

No separate preprocessing step is required beyond dataset conversion (documented above). The conversion scripts are embedded in the notebook cells and produce JSON files in the format expected by Search-o1's runner scripts (`{'Question': str, 'Correct Choice': str}` for GPQA; `{'question': str, 'answer': str/list}` for NQ/HotpotQA).

---

## Running Experiments (Inference)

**No training is involved.** All experiments are inference-only using pretrained models.

### Models Used

| Model | HuggingFace ID | Role |
|-------|----------------|------|
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Primary (all datasets) |
| QwQ-32B-Preview (AWQ) | `Qwen/QwQ-32B-Preview-AWQ` | Secondary (GPQA only, separate notebook) |

### Search API Setup

We use [Brave Search API](https://brave.com/search/api/) (free tier: ~1,000 queries/month per account) as a replacement for the original paper's Bing Search API, which has been retired.

```bash
export BRAVE_KEY="your_brave_api_key_here"
```

### Example Commands

All commands are run from the `Search-o1/scripts/` directory. The notebook documents the exact commands used, but here are representative examples:

**Direct Reasoning:**
```bash
python run_direct_gen.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --temperature 0.7 --top_p 0.8 --top_k 20 \
    --repetition_penalty 1.05 --max_tokens 16384
```

**Standard RAG (Naive RAG):**
```bash
python run_naive_rag.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --bing_subscription_key $BRAVE_KEY \
    --top_k 10 --max_doc_len 3000 \
    --temperature 0.7 --top_p 0.8 \
    --repetition_penalty 1.05 --max_tokens 16384
```

**RAG Agent:**
```bash
python run_rag_agent.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --bing_subscription_key $BRAVE_KEY \
    --max_search_limit 5 --max_url_fetch 5 --max_turn 10 --top_k 10 \
    --temperature 0.7 --top_p 0.8 \
    --repetition_penalty 1.05 --max_tokens 16384
```

**Search-o1:**
```bash
python run_search_o1.py \
    --dataset_name gpqa \
    --split diamond \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --bing_subscription_key $BRAVE_KEY \
    --max_search_limit 5 --max_turn 10 --top_k 10 --max_doc_len 3000 \
    --temperature 0.7 --top_p 0.8 \
    --repetition_penalty 1.05 --max_tokens 16384
```

Replace `--dataset_name` and `--split` for other datasets:
- NQ: `--dataset_name nq --split test`
- HotpotQA: `--dataset_name hotpotqa --split test`

---

## Evaluation

Evaluation uses the original `evaluate.py` with one patch: a fallback regex `ANSWER:\s*([A-D])` to handle cases where the 7B model outputs `ANSWER: B` instead of `\boxed{B}`.

```bash
python evaluate.py --output_path outputs/<path_to_output>.json
```

This produces a `.metrics.json` file alongside the output. To aggregate all results:

```python
import json, glob
for f in sorted(glob.glob('outputs/**/*.metrics.json', recursive=True)):
    with open(f) as fh:
        data = json.load(fh)
    overall = data['overall']
    print(f"{f}")
    print(f"  em={overall['em']:.3f}  f1={overall.get('f1', 0):.3f}  valid={overall['num_valid_answer']}")
```

---

## Pretrained Models

No fine-tuning or training was performed. We used the following publicly available pretrained models directly from HuggingFace:

- [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [`Qwen/QwQ-32B-Preview-AWQ`](https://huggingface.co/Qwen/QwQ-32B-Preview-AWQ)

---

## Key Modifications from Original Code

1. **`bing_search.py` → Brave Search API:** The original Bing Search API has been retired. We created a drop-in replacement using Brave Search API that preserves all original function signatures. The notebook documents 4 iterative versions of this replacement (Cells 17, 18, 54, 60).

2. **Snippet-based retrieval instead of Jina Reader:** The original paper uses Jina Reader to fetch full web page content. We use Brave Search snippets directly, which returns shorter but more focused text.

3. **`evaluate.py` fallback regex:** Added `ANSWER:\s*([A-D])` pattern alongside the original `\boxed{}` extraction to handle the 7B model's output format (Cell 45).

4. **`pyext` dummy module:** The original code imports `pyext` for LiveCodeBench. Since we don't use that dataset, we install a dummy module to prevent import errors (Cell 39).

5. **Token limit:** We used `--max_tokens 16384` (vs. the paper's 32768) due to memory constraints.

---

## Hyperparameters

| Parameter            | Value  | Note                                    |
|----------------------|--------|-----------------------------------------|
| temperature          | 0.7    | Same as original paper                  |
| top_p                | 0.8    | Same as original paper                  |
| top_k                | 20     | Same as original paper (Direct only)    |
| repetition_penalty   | 1.05   | Same as original paper                  |
| max_tokens           | 16384  | Halved from paper's 32768               |
| max_search_limit     | 5–10   | Per-dataset (GPQA: 5, HotpotQA: 10)    |
| max_turn             | 10–15  | Per-dataset (GPQA: 10, HotpotQA: 15)   |
| top_k (search)       | 10     | Number of search results retrieved      |
| max_doc_len          | 3000   | Max document length for RAG context     |

---

## Computational Requirements

- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM); Google Colab A100 (earlier runs)
- **7B model:** ~2–4 hours per dataset×method combination
- **32B model (AWQ):** ~6–12 hours per dataset×method combination on GPQA
- **Total compute:** Approximately 80–100 GPU-hours across all experiments

See the full report for detailed pre-experiment estimates vs. actual requirements.

---

## License

This project is for academic purposes (CSE 517 coursework). The original Search-o1 code is subject to its own license — see the [original repository](https://github.com/sunnynexus/Search-o1).

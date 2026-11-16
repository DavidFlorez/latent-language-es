# Experiment Notes: `latent-language-es`

## Goal
Test whether small open-weight multilingual LLMs (`1B–2B` parameters) exhibit an English "pivot" representation in intermediate layers when translating common concrete nouns from Spanish into Catalan (`es → ca`) and French (`es → fr`) without any English in the prompt, using the logit lens and an L2-normalized `lm_head` cosine-lens control.

## Setup
- **Dataset:** `data/words.csv` — 150 common concrete nouns in `es`, `ca`, `fr`, `en`, `zh` (all rows satisfy `es != ca`, `es != fr`, `en != es/ca/fr`, and `en[0]` differs from `es[0]`, `ca[0]`, `fr[0]` to avoid trivial 1-character prefix overlap with English).
- **Prompts:** 5-shot Wendler-style completion with zero English (`Español: "<es>" - Català: "<ca>"` / `Français: "<fr>"`), deterministic sampling (`seed=42`, excluding query row).
- **Models:** `Qwen/Qwen2.5-1.5B`, `BSC-LT/salamandra-2b`, `utter-project/EuroLLM-1.7B`, `meta-llama/Llama-3.2-1B` (gated; requires `HF_TOKEN`).
- **Execution environment:** Google Colab GPU runtime (T4, CUDA, `torch.float16`, one model at a time, freeing memory between models). Exact versions and commands recorded as runs execute.

## Experiment Log

### 2026-11-16
- **Milestone 1 (Data & Repo Init):** Created 150-noun dataset `data/words.csv`, `.gitignore`, `requirements.txt`, `README.md`, and `NOTES.md`. Verified all 150 rows have unique entries per language, zero `es==ca` or `es==fr` identity copies, and zero first-letter collisions between `en` and `es`/`ca`/`fr`.

## Key Results
_Pending Colab GPU runs (`results/summary.csv`)._

## Visuals
_Pending Colab GPU runs (`figures/`)._

## Surprises and Limitations
_To be updated after inspecting logit lens, cosine lens, and top-5 decoded token outputs._

## Next Steps
- Implement `prompts.py`, `logit_lens.py`, `plotting.py`, and `run.py`.
- Push to GitHub (`DavidFlorez/latent-language-es`), provision Colab T4 GPU runtime, run tokenizer checks and experiments, and copy `figures/` and `results/` back into the repo.

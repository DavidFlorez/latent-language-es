# latent-language-es

Testing whether small multilingual LLMs use English as a hidden "pivot" (latent) language when translating from Spanish (`es → ca` and `es → fr`), using the logit lens and a cosine-lens control.

Inspired by Wendler et al. (2024), *"Do Llamas Work in English? On the Latent Language of Multilingual Transformers"* ([epfl-dlab/llm-latent-language](https://github.com/epfl-dlab/llm-latent-language)).

## Repository Structure

- `data/words.csv`: 150 common concrete nouns with translations in Spanish (`es`), Catalan (`ca`), French (`fr`), English (`en`), and Simplified Chinese (`zh`), curated to avoid English cognates and cross-language first-letter collisions with `en`.
- `prompts.py`: Wendler-style 5-shot translation prompt builder with zero English anywhere (`es → ca` and `es → fr`).
- `logit_lens.py`: Model loading (CUDA fp16), tokenizer single-token analysis, prefix-token probability extraction (with and without leading space), standard logit lens, L2-normalized `lm_head` cosine lens control, and top-5 decoded token inspection.
- `plotting.py`: Per-model line plots (mean ± 95% CI across layers) and combined multi-model comparison figure.
- `run.py`: Single CLI entry point (`python run.py --model <id> --direction es-ca`).
- `figures/`: Per-model and combined figures copied back from the Google Colab T4 runtime.
- `results/`: `summary.csv`, tokenizer single-token statistics, per-layer probability arrays, and top-5 decoded token tables.
- `NOTES.md`: Experiment log, setup details, dated attempt/error/fix history, quantitative findings, and limitations.

## Setup (Google Colab GPU Runtime)

```bash
git clone https://github.com/DavidFlorez/latent-language-es.git
cd latent-language-es
pip install -r requirements.txt
```

For gated models (`meta-llama/Llama-3.2-1B`), export `HF_TOKEN` or store it in Colab Secrets as `HF_TOKEN`.

## Usage

Check tokenizer single-token counts across all languages:

```bash
python run.py --check-tokenizers
```

Run logit lens + cosine lens for a model and direction:

```bash
python run.py --model Qwen/Qwen2.5-1.5B --direction es-ca
python run.py --model Qwen/Qwen2.5-1.5B --direction es-fr
```

Models evaluated (executed one at a time in fp16 on Colab T4, freeing CUDA memory between runs):
- `Qwen/Qwen2.5-1.5B`
- `BSC-LT/salamandra-2b`
- `utter-project/EuroLLM-1.7B`
- `meta-llama/Llama-3.2-1B` (skipped with warning if `HF_TOKEN` is missing)

Rebuild the combined comparison figure across all completed runs:

```bash
python run.py --combine-only
```

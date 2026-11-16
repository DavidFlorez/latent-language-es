# Experiment Notes: `latent-language-es`

## Goal
Test whether small open-weight multilingual LLMs (`1B–2B` parameters) use English as a hidden "pivot" (latent) language when translating common concrete nouns from Spanish into Catalan (`es → ca`) and French (`es → fr`) without any English in the prompt, using the logit lens and an L2-normalized `lm_head` cosine-lens control.

## Setup
- **Dataset:** `data/words.csv` — 150 common concrete nouns in `es`, `ca`, `fr`, `en`, `zh` (all rows have 150 unique entries per language, `es != ca`, `es != fr`, `en != es/ca/fr`, and zero first-letter collisions between `en` and `es`/`ca`/`fr`; initial `'s'` occurs in `26/150` (`17.3%`) `en` words vs `4/150` (`2.7%`) `es`, `2/150` (`1.3%`) `ca`, and `3/150` (`2.0%`) `fr`).
- **Prompts:** 5-shot Wendler-style completion with zero English anywhere (`Español: "<es>" - Català: "` or `Français: "`), deterministic few-shot sampling (`seed=42`, excluding query row).
- **Models:** `Qwen/Qwen2.5-1.5B` (28 layers), `BSC-LT/salamandra-2b` (24 layers), `utter-project/EuroLLM-1.7B` (24 layers), and `meta-llama/Llama-3.2-1B` (gated; skipped with warning when `HF_TOKEN` is absent).
- **Execution environment:** Google Colab GPU runtime (`Tesla T4`, `15360 MiB` VRAM), Python `3.13.16`, `torch 2.14.1+cu130`, `transformers 5.22.1`, `accelerate 1.15.0`, `huggingface_hub 1.33.0`, `sentencepiece 0.2.2`, `scipy 1.16.3`, `matplotlib 3.10.0`, `pandas 2.2.3`, `numpy 2.4.6`.
- **Precision & memory discipline:** CUDA `torch.float16`, one model at a time in its own subprocess (`release_model()` + `torch.cuda.empty_cache()` between runs).
- **Exact commands run on Colab:**
  ```bash
  python run.py --model Qwen/Qwen2.5-1.5B --direction both
  python run.py --model BSC-LT/salamandra-2b --direction both
  python run.py --model utter-project/EuroLLM-1.7B --direction both
  python run.py --model meta-llama/Llama-3.2-1B --direction es-ca
  python run.py --check-tokenizers --direction both
  python run.py --combine-only
  ```

## Experiment Log

### 2026-11-16
- **Milestone 1 (Data & Repo Init):** Created 150-noun dataset `data/words.csv`, `.gitignore`, `requirements.txt`, `README.md`, and `NOTES.md`. Verified zero identity copies (`es==ca`, `es==fr`, `en==es/ca/fr`) and zero first-letter `en` collisions. Committed (`148a752`).
- **Milestone 2 (Core Pipeline):** Implemented `prompts.py`, `logit_lens.py`, `plotting.py`, and `run.py`, including anchor-based (`"Hello"`) batch vocabulary decoding preserving leading spaces across SentencePiece and byte-level BPE, double-norm verification against `model(**inputs).logits`, L2-normalized `lm_head` cosine-lens control, 95% t-interval CIs, top-5 decoded token tables, and gated HF repo fallback. Pushed to `https://github.com/DavidFlorez/latent-language-es.git` (`0969d08`).
- **Colab Provisioning & Tooling Fixes:**
  - Initial `colab new` hit transient HTTP 503 on T4 allocation; succeeded on retry (`latent-lang`, `Tesla T4`).
  - `colab` CLI hit `WebSocketConnectionClosedException: SSL: CERTIFICATE_VERIFY_FAILED` under local Python 3.12 (`cafile=None`); resolved by setting `SSL_CERT_FILE=/etc/ssl/cert.pem`.
  - Confirmed `google.colab.userdata.get("HF_TOKEN")` times out in headless `colab exec`; `get_hf_token()` catches the timeout and cleanly skips `meta-llama/Llama-3.2-1B` with a warning.
  - Verified on Colab that `outputs.hidden_states[-1]` in `transformers 5.22.1` already applies `final_norm` across all three models (`detect_last_hidden_already_normed`: direct vs re-normed logit max abs diff was `0.007812` vs `10.281250` on Qwen2.5-1.5B, `0.007812` vs `1.667969` on salamandra-2b, and `0.007812` vs `11.044922` on EuroLLM-1.7B).
- **Smoke Test & CUDA OOM Fix (`4e1d1a5`):**
  - 5-word smoke test on `Qwen/Qwen2.5-1.5B` succeeded and confirmed greedy outputs (`oveja → ovella`, `caballo → caball`, `perro → còrsega`) and late-layer `' dog'`/`'狗'`/`' cheese'` activations.
  - First full run on `BSC-LT/salamandra-2b` (`256,000`-token vocab) hit `torch.OutOfMemoryError` trying to allocate a second `1.95 GiB` `float32` unembedding matrix in `F.normalize(w_float, p=2, dim=-1)` (alongside `6.29 GiB` stale IPython kernel VRAM).
  - Updated `logit_lens.py` (`4e1d1a5`) to compute `lm_head` row L2 norms in `32,768`-row chunks (`~1 MB` final vector) and scale `(logits_std - bias) / row_norms` directly without duplicating `lm_head.weight` in `float32`, cleared kernel VRAM (`%reset -f`), and verified identical `Qwen/Qwen2.5-1.5B` results (`0.3409` / `0.4415` cosine Peak EN).
- **Full Runs & Artifact Sync:** Executed all three accessible models across all 150 nouns (`es-ca` and `es-fr`), confirmed `meta-llama/Llama-3.2-1B` gated skip, regenerated `results/tokenizer_stats.csv` and combined figures, copied `figures/` and `results/` back into the repo, and terminated Colab session `latent-lang`.

## Key Results

### Summary Metrics (`results/summary.csv`, `n=150` nouns per direction)

| Model | Dir | Layers | Logit Peak EN (Layer / Depth%) | Logit Final Target | Logit Final Source | Top-1 Acc | Cosine Peak EN (Layer / Depth%) | Cosine Final Target | Mean `||W||_2` (`es`/`ca`/`fr`/`en`/`zh`) |
|---|---|---:|---|---:|---:|---:|---|---:|---|
| `Qwen/Qwen2.5-1.5B` | `es-ca` | 28 | `0.2602` (`L25` / `89.3%`) | `0.2846` | `0.4286` | `0.527` | `0.3409` (`L25` / `89.3%`) | `0.2062` | `1.063 / 1.065 / 1.057 / 1.061 / 1.060` |
| `Qwen/Qwen2.5-1.5B` | `es-fr` | 28 | `0.3641` (`L24` / `85.7%`) | `0.5169` | `0.2319` | `0.700` | `0.4415` (`L25` / `89.3%`) | `0.4474` | `1.063 / 1.065 / 1.057 / 1.061 / 1.060` |
| `BSC-LT/salamandra-2b` | `es-ca` | 24 | `0.0850` (`L21` / `87.5%`) | `0.5267` | `0.3426` | `0.673` | `0.1774` (`L20` / `83.3%`) | `0.5089` | `0.861 / 0.864 / 0.860 / 0.852 / —` |
| `BSC-LT/salamandra-2b` | `es-fr` | 24 | `0.1268` (`L21` / `87.5%`) | `0.5885` | `0.1981` | `0.740` | `0.2137` (`L19` / `79.2%`) | `0.5441` | `0.861 / 0.864 / 0.860 / 0.852 / —` |
| `utter-project/EuroLLM-1.7B` | `es-ca` | 24 | `0.2356` (`L20` / `83.3%`) | `0.4562` | `0.4782` | `0.620` | `0.2006` (`L20` / `83.3%`) | `0.2429` | `1.250 / 1.250 / 1.252 / 1.256 / 1.364` |
| `utter-project/EuroLLM-1.7B` | `es-fr` | 24 | `0.3113` (`L20` / `83.3%`) | `0.7002` | `0.2295` | `0.827` | `0.2593` (`L20` / `83.3%`) | `0.4047` | `1.250 / 1.250 / 1.252 / 1.256 / 1.364` |

### Tokenizer Single-Token Coverage (`results/tokenizer_stats.csv`, out of 150)

| Model | Vocab | `es` either (no/with sp) | `ca` either | `fr` either | `en` either | `zh` either | Post-quote `1`-tok (`es`/`ca`/`fr`/`en`/`zh`) |
|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-1.5B` | 151,665 | `27` (`18.0%`, `10`/`23`) | `41` (`27.3%`) | `42` (`28.0%`) | `142` (`94.7%`) | `117` (`78.0%`) | `10 / 29 / 24 / 101 / 117` |
| `BSC-LT/salamandra-2b` | 256,000 | `73` (`48.7%`) | `77` (`51.3%`) | `62` (`41.3%`) | `104` (`69.3%`) | `0` (`0.0%`, mean `6.10` toks) | `20 / 46 / 42 / 94 / 0` |
| `utter-project/EuroLLM-1.7B` | 128,000 | `65` (`43.3%`) | `54` (`36.0%`) | `58` (`38.7%`) | `116` (`77.3%`) | `2` (`1.3%`, mean `2.53` toks) | `14 / 35 / 33 / 83 / 59` |
| `meta-llama/Llama-3.2-1B` | — | skipped (`HF_TOKEN` missing) | — | — | — | — | — |

### Top-5 Decoded Token Highlights (`results/top5_examples.md`)
- **`Qwen/Qwen2.5-1.5B`:** At layers `22–26` (`~79–93%` depth), top tokens simultaneously surface English and Chinese concepts before decoding into the target language: e.g., `perro → chien` (`es-fr`) hits `'狗'` (`0.739` at L25) and `' dog'` (`0.165`; cosine L25 `' dog'` `0.690`, `'狗'` `0.233`), then transitions cleanly at L27–28 to `'ch'` (`0.959` → `0.940`); `queso → fromage` hits `' cheese'` (`0.702–0.831` at L24–26) before `'from'` (`0.517` at L28). In `es-ca`, weaker Catalan knowledge leads to source interference at L28 (`'ques'` `0.296` for `queso → formatge`, `'can'` `0.349` for `perro → gos`).
- **`BSC-LT/salamandra-2b`:** Despite ~3x smaller aggregate English peak (`0.085` `es-ca`, `0.127` `es-fr`), non-cognates still strongly activate English at layers `19–20`: `perro → gos` peaks on `' dogs'` (`0.533` logit, `0.996` cosine at L20) before locking onto `'gos'` (`0.929` at L24); `manzana → poma` peaks on `' fruit'` (`0.611` logit, `0.994` cosine at L19) before `' poma'` (`0.141` at L24); `queso → formatge` hits `' cheese'` (`0.871` at L19) before `'ques'` (`0.452` at L24).
- **`utter-project/EuroLLM-1.7B`:** Strong L18–21 English surge (`Peak EN` at L20: `0.236` `es-ca`, `0.311` `es-fr`), with `queso → formatge` reaching `' cheese'` `1.000` at L20 (both logit and cosine lens), `manzana → poma` peaking on `'fruit'` (`0.888` at L19) / `' maç'` (Portuguese interference `0.499` at L21) / `'man'` (`0.715` at L24), and `perro → gos` activating `'犬'` (`0.695` at L19) and Russian `' соба'` (`0.460` cosine at L20) alongside `' dogs'` before `'g'` (`0.753` at L24).

## Visuals
- Combined cross-model logit lens comparison: [`figures/combined_comparison.png`](figures/combined_comparison.png)
- Combined cross-model cosine lens comparison: [`figures/combined_cosine_comparison.png`](figures/combined_cosine_comparison.png)
- `Qwen/Qwen2.5-1.5B`: [`figures/Qwen2.5-1.5B_es-ca.png`](figures/Qwen2.5-1.5B_es-ca.png) | [`figures/Qwen2.5-1.5B_es-fr.png`](figures/Qwen2.5-1.5B_es-fr.png)
- `BSC-LT/salamandra-2b`: [`figures/salamandra-2b_es-ca.png`](figures/salamandra-2b_es-ca.png) | [`figures/salamandra-2b_es-fr.png`](figures/salamandra-2b_es-fr.png)
- `utter-project/EuroLLM-1.7B`: [`figures/EuroLLM-1.7B_es-ca.png`](figures/EuroLLM-1.7B_es-ca.png) | [`figures/EuroLLM-1.7B_es-fr.png`](figures/EuroLLM-1.7B_es-fr.png)

## Surprises and Limitations
1. **Bilingual EN+ZH Conceptual Pivot in Qwen2.5-1.5B:** Rather than pivoting purely in English, `Qwen/Qwen2.5-1.5B` jointly activates English (`0.260–0.364`) and Chinese (`~0.21–0.26`) semantic tokens in layers 22–26, even though the prompt contains zero Chinese or English.
2. **Cosine-Lens Control Exposes Opposite Norm Biases Across Architectures:**
   - On **`Qwen/Qwen2.5-1.5B`**, layers 2–9 show an artificial `~0.16–0.17` shallow English plateau because early hidden states project heavily onto raw `'s'` (`0.81–0.98` probability), and `17.3%` (`26/150`) of `en` nouns start with `'s'` vs `1.3–2.7%` in `es`/`ca`/`fr`. L2-normalizing `lm_head` drops this shallow artifact 3–4x (`~0.04–0.05`) while *increasing* the true conceptual L25 English peak (`0.260 → 0.341` `es-ca`, `0.364 → 0.442` `es-fr`).
   - On **`BSC-LT/salamandra-2b`**, `en` unembedding rows have slightly smaller mean L2 norm (`0.8515`) than `ca` (`0.8636`), `es` (`0.8605`), and `fr` (`0.8598`). Standard logit lens therefore understates Salamandra's internal English activation (`0.085` `es-ca`, `0.127` `es-fr`), whereas the cosine lens doubles it (`0.177` / `0.214` at L19–20). Even after norm correction, however, Salamandra retains less than half the English pivot peak of Qwen2.5-1.5B and achieves strong Iberian translation (`0.673` `es-ca` Top-1 Acc).
   - On **`utter-project/EuroLLM-1.7B`**, `lm_head` norms are larger (`1.25–1.36`), so cosine normalization lowers both peak EN (`0.236 → 0.201`, `0.311 → 0.259`) and final target calibration (`0.700 → 0.405` `es-fr`) while preserving the L20 English pivot peak.
3. **SentencePiece Leading-Space & Byte-Fallback Quirks:** Standalone `tokenizer.encode("狗")` in SentencePiece (`EuroLLM-1.7B`) inserts a bare `'▁'` token (`['▁', '狗']`), yielding `2/150` standalone single-token `zh` words even though `59/150` are single-token immediately after `"` (`prompt + "狗"`). Conversely, `BSC-LT/salamandra-2b` has zero CJK character tokens (`0/150`; UTF-8 byte fallback `<0xE7>..`), so its `zh` curve stays at `0.000` under valid-UTF-8 prefix matching.
4. **Source Interference on `es → ca` & Single-Char Prefixes:** All three models exhibit notable Spanish-copy interference on `es → ca` (`final_source_prob` = `0.429` Qwen, `0.343` Salamandra, `0.478` EuroLLM vs `0.198–0.232` on `es → fr`), and prefix-matching allows 1-letter subwords (`'p'`, `'g'`, `'c'`) to capture generic uncertainty near the final layer.
5. **Gated Baseline Missing:** `meta-llama/Llama-3.2-1B` was skipped because `HF_TOKEN` was not provided.

## Next Steps
- Supply `HF_TOKEN` in a Colab runtime and run `python run.py --model meta-llama/Llama-3.2-1B --direction both && python run.py --combine-only` to include Llama-3.2-1B in `results/summary.csv` and the combined plots.
- Split `data/words.csv` into close Romance cognates vs non-cognates (e.g., `perro/gos` vs `caballo/cavall`) to measure whether direct `es → ca` paths in `BSC-LT/salamandra-2b` are driven primarily by cognate pairs or hold equally on non-cognates.

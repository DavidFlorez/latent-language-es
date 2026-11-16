"""Single CLI entry point for running logit lens translation experiments on Colab GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from logit_lens import (
    GatedModelSkipError,
    check_tokenizers,
    get_hf_token,
    load_model_and_tokenizer,
    release_model,
    run_logit_lens,
)
from plotting import build_combined_figures, compute_mean_ci, plot_model_direction
from prompts import ALL_LANGS, load_words, normalize_direction, parse_direction

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "BSC-LT/salamandra-2b",
    "utter-project/EuroLLM-1.7B",
    "meta-llama/Llama-3.2-1B",
]


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _summarize_run(out, probs_arr: np.ndarray, prefix: str = "") -> dict[str, Any]:
    src_lang, tgt_lang = parse_direction(out.direction)
    en_idx = out.langs.index("en")
    tgt_idx = out.langs.index(tgt_lang)
    src_idx = out.langs.index(src_lang)

    mean_by_layer = probs_arr.mean(axis=0)  # (L, 5)
    peak_en_idx = int(np.argmax(mean_by_layer[:, en_idx]))
    peak_tgt_idx = int(np.argmax(mean_by_layer[:, tgt_idx]))

    return {
        f"{prefix}peak_en_layer": peak_en_idx + 1,
        f"{prefix}peak_en_depth_pct": round(100.0 * (peak_en_idx + 1) / out.num_layers, 1),
        f"{prefix}peak_en_prob": round(float(mean_by_layer[peak_en_idx, en_idx]), 6),
        f"{prefix}final_target_prob": round(float(mean_by_layer[-1, tgt_idx]), 6),
        f"{prefix}final_en_prob": round(float(mean_by_layer[-1, en_idx]), 6),
        f"{prefix}final_source_prob": round(float(mean_by_layer[-1, src_idx]), 6),
        f"{prefix}peak_target_layer": peak_tgt_idx + 1,
        f"{prefix}peak_target_prob": round(float(mean_by_layer[peak_tgt_idx, tgt_idx]), 6),
    }


def _save_layer_stats(out, results_dir: Path) -> Path:
    slug = _model_slug(out.model_id)
    rows = []
    for lens_name, arr in (("logit_lens", out.probs_logit), ("cosine_lens", out.probs_cosine)):
        for lang_i, lang in enumerate(out.langs):
            mean, lower, upper = compute_mean_ci(arr[:, :, lang_i])
            for l_idx in range(out.num_layers):
                rows.append(
                    {
                        "model": out.model_id,
                        "direction": out.direction,
                        "lens": lens_name,
                        "layer": l_idx + 1,
                        "lang": lang,
                        "mean": round(float(mean[l_idx]), 6),
                        "ci95_low": round(float(lower[l_idx]), 6),
                        "ci95_high": round(float(upper[l_idx]), 6),
                    }
                )
    path = results_dir / f"layer_stats_{slug}_{out.direction}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _upsert_summary(summary_row: dict[str, Any], summary_path: Path) -> pd.DataFrame:
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        mask = ~(
            (df["model"] == summary_row["model"]) & (df["direction"] == summary_row["direction"])
        )
        df = pd.concat([df[mask], pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    df.to_csv(summary_path, index=False)
    return df


def _rebuild_top5_markdown(results_dir: Path) -> Path | None:
    csv_files = sorted(results_dir.glob("top5_*.csv"))
    if not csv_files:
        return None

    lines = ["# Top-5 Decoded Tokens per Layer (3 Example Words)\n"]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        for (model_id, direction, lens, word_es, word_tgt, word_en), grp in df.groupby(
            ["model", "direction", "lens", "word_es", "word_target", "word_en"], sort=False
        ):
            lines.append(
                f"## `{model_id}` | `{direction}` | `{lens}` | "
                f"`{word_es}` → `{word_tgt}` (en: `{word_en}`)\n"
            )
            lines.append("| Layer | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 |")
            lines.append("|---:|---|---|---|---|---|")
            for _, r in grp.sort_values("layer").iterrows():
                cells = [
                    f"`{r[f'top{k}_token']}` ({r[f'top{k}_prob']:.3f})" for k in range(1, 6)
                ]
                lines.append(f"| {int(r['layer'])} | " + " | ".join(cells) + " |")
            lines.append("")

    out_md = results_dir / "top5_examples.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def _save_run_outputs(out, results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(out.model_id)

    # Save raw probability tensors for fast replotting/combination.
    npz_path = results_dir / f"probs_{slug}_{out.direction}.npz"
    np.savez_compressed(
        npz_path,
        probs_logit=out.probs_logit,
        probs_cosine=out.probs_cosine,
        langs=np.array(out.langs),
    )

    _save_layer_stats(out, results_dir)

    if out.top5_records:
        top5_path = results_dir / f"top5_{slug}_{out.direction}.csv"
        pd.DataFrame(out.top5_records).to_csv(top5_path, index=False)

    summary_row: dict[str, Any] = {
        "model": out.model_id,
        "direction": out.direction,
        "n_words": out.n_words,
        "num_layers": out.num_layers,
    }
    summary_row.update(_summarize_run(out, out.probs_logit, prefix=""))
    summary_row["target_top1_acc"] = round(out.top1_acc_logit, 4)
    summary_row.update(_summarize_run(out, out.probs_cosine, prefix="cosine_"))
    summary_row["cosine_target_top1_acc"] = round(out.top1_acc_cosine, 4)
    for lang in ALL_LANGS:
        summary_row[f"mean_norm_{lang}"] = out.lm_head_norm_stats.get(lang)

    _upsert_summary(summary_row, results_dir / "summary.csv")

    fig_path = figures_dir / f"{slug}_{out.direction}.png"
    plot_model_direction(
        out.probs_logit,
        out.probs_cosine,
        out.langs,
        out.model_id,
        out.direction,
        fig_path,
    )

    print(
        f"[Result] {out.model_id} [{out.direction}] | "
        f"Logit Peak EN: L{summary_row['peak_en_layer']}/{out.num_layers} "
        f"({summary_row['peak_en_prob']:.4f}), Final Target: {summary_row['final_target_prob']:.4f}, "
        f"Top-1 Acc: {summary_row['target_top1_acc']:.3f} | "
        f"Cosine Peak EN: L{summary_row['cosine_peak_en_layer']}/{out.num_layers} "
        f"({summary_row['cosine_peak_en_prob']:.4f}), Final Target: {summary_row['cosine_final_target_prob']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run logit-lens multilingual pivot experiment (es->ca, es->fr)."
    )
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model ID.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional list of models (used with --all-models or --check-tokenizers).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all default models sequentially, one at a time, freeing CUDA memory between models.",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="es-ca",
        help="Translation direction: 'es-ca', 'es-fr', or 'both'.",
    )
    parser.add_argument("--data", type=str, default="data/words.csv", help="Path to words CSV.")
    parser.add_argument("--results-dir", type=str, default="results", help="Results directory.")
    parser.add_argument("--figures-dir", type=str, default="figures", help="Figures directory.")
    parser.add_argument("--device", type=str, default="cuda", help="Device ('cuda' on Colab).")
    parser.add_argument("--num-shots", type=int, default=5, help="Few-shot examples per prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for few-shot sampling.")
    parser.add_argument(
        "--example-words",
        nargs="+",
        default=["perro", "manzana", "queso"],
        help="3 Spanish words for top-5 decoded tokens per layer.",
    )
    parser.add_argument(
        "--check-tokenizers",
        action="store_true",
        help="Check single-token counts per language across model tokenizers.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Rebuild combined comparison figures and top-5 markdown from saved results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional word limit for quick smoke tests.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.combine_only:
        created = build_combined_figures(results_dir, figures_dir)
        md_path = _rebuild_top5_markdown(results_dir)
        print(f"[Combined] Built {len(created)} combined figures; top5 md={md_path}")
        return 0

    df = load_words(args.data)
    if args.limit is not None:
        # Ensure example words remain included during smoke tests when possible
        ex_mask = df["es"].isin(args.example_words)
        head_df = df.iloc[: args.limit]
        df = pd.concat([head_df, df[ex_mask]]).drop_duplicates().reset_index(drop=True)

    hf_token = get_hf_token()
    print(f"[Env] HF_TOKEN available: {bool(hf_token)}")

    if args.check_tokenizers:
        model_list = args.models or ([args.model] if args.model else DEFAULT_MODELS)
        tok_df = check_tokenizers(df, model_list, hf_token=hf_token)
        tok_path = results_dir / "tokenizer_stats.csv"
        tok_df.to_csv(tok_path, index=False)
        print(f"[Tokenizer Check] Saved {tok_path}")
        if not tok_df.empty:
            print(tok_df.to_string(index=False))
        if not args.model and not args.all_models:
            return 0

    if args.all_models:
        model_list = args.models or DEFAULT_MODELS
    elif args.model:
        model_list = [args.model]
    else:
        parser.error("Specify --model <id>, --all-models, --check-tokenizers, or --combine-only.")
        return 2

    if args.direction.strip().lower() == "both":
        directions = ["es-ca", "es-fr"]
    else:
        directions = [normalize_direction(args.direction)]

    # Save runtime metadata for reproducibility
    meta = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "n_words": len(df),
        "directions": directions,
        "models": model_list,
    }
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:
        pass
    (results_dir / "runtime_env.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for model_id in model_list:
        print(f"\n=== Loading {model_id} (fp16 on {args.device}) ===")
        model = None
        tokenizer = None
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_id,
                device=args.device,
                dtype=torch.float16,
                hf_token=hf_token,
            )
        except GatedModelSkipError as exc:
            print(f"[WARNING] {exc} Skipping {model_id}.")
            continue

        try:
            for direction in directions:
                out = run_logit_lens(
                    model=model,
                    tokenizer=tokenizer,
                    model_id=model_id,
                    df=df,
                    direction=direction,
                    device=args.device,
                    num_shots=args.num_shots,
                    seed=args.seed,
                    example_words=tuple(args.example_words),
                )
                _save_run_outputs(out, results_dir, figures_dir)
        finally:
            print(f"=== Releasing {model_id} from memory ===")
            release_model(model, tokenizer)

    created = build_combined_figures(results_dir, figures_dir)
    md_path = _rebuild_top5_markdown(results_dir)
    print(f"\n[Done] Combined figures: {[str(p) for p in created]}; top-5 table: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

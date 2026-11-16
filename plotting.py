"""Plotting utilities for per-model and multi-model logit lens comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from prompts import ALL_LANGS, parse_direction

LANG_STYLES = {
    "es": {"color": "#1f77b4", "label": "es (Spanish)", "ls": "-", "lw": 2.0},
    "ca": {"color": "#e66101", "label": "ca (Catalan)", "ls": "-", "lw": 2.1},
    "fr": {"color": "#2ca02c", "label": "fr (French)", "ls": "-", "lw": 2.1},
    "en": {"color": "#d62728", "label": "en (English)", "ls": "-", "lw": 2.4},
    "zh": {"color": "#7f7f7f", "label": "zh (Chinese)", "ls": "--", "lw": 1.6},
}


def compute_mean_ci(probs_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and 95% two-sided t-interval across words (axis=0)."""
    n = probs_2d.shape[0]
    mean = probs_2d.mean(axis=0)
    if n <= 1:
        return mean, mean, mean
    sem = probs_2d.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    half = t_crit * sem
    lower = np.clip(mean - half, 0.0, 1.0)
    upper = np.clip(mean + half, 0.0, 1.0)
    return mean, lower, upper


def _plot_panel(
    ax: plt.Axes,
    probs_3d: np.ndarray,
    langs: list[str],
    tgt_lang: str,
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Plot mean ± 95% CI trajectories for all languages on a single matplotlib Axes."""
    num_layers = probs_3d.shape[1]
    layers = np.arange(1, num_layers + 1)

    en_idx = langs.index("en")
    en_mean = probs_3d[:, :, en_idx].mean(axis=0)
    peak_en_idx = int(np.argmax(en_mean))
    peak_en_layer = peak_en_idx + 1
    peak_en_val = float(en_mean[peak_en_idx])

    for i, lang in enumerate(langs):
        style = LANG_STYLES[lang]
        mean, lower, upper = compute_mean_ci(probs_3d[:, :, i])
        label = style["label"] + (" [target]" if lang == tgt_lang else "")
        ax.plot(
            layers,
            mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            label=label,
        )
        ax.fill_between(layers, lower, upper, color=style["color"], alpha=0.18)

    ax.axvline(peak_en_layer, color=LANG_STYLES["en"]["color"], linestyle=":", alpha=0.55, lw=1.3)
    ax.annotate(
        f"Peak EN: L{peak_en_layer} ({peak_en_val:.3f})",
        xy=(peak_en_layer, peak_en_val),
        xytext=(0.03, 0.92),
        textcoords="axes fraction",
        fontsize=8.5,
        color=LANG_STYLES["en"]["color"],
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=LANG_STYLES["en"]["color"], alpha=0.85),
    )

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Layer", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Probability (mean ± 95% CI)", fontsize=10)
    ax.set_xlim(1, num_layers)
    ax.set_ylim(0.0, None)
    step = max(2, num_layers // 7)
    xticks = list(range(1, num_layers + 1, step))
    if num_layers not in xticks:
        xticks.append(num_layers)
    ax.set_xticks(xticks)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_direction(
    probs_logit: np.ndarray,
    probs_cosine: np.ndarray,
    langs: list[str],
    model_id: str,
    direction: str,
    out_path: str | Path,
) -> Path:
    """Save a side-by-side Logit Lens vs Cosine Lens figure for one model and direction."""
    src_lang, tgt_lang = parse_direction(direction)
    n_words = probs_logit.shape[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180, sharey=False)
    _plot_panel(
        axes[0],
        probs_logit,
        langs,
        tgt_lang=tgt_lang,
        title="Standard Logit Lens (final norm + lm_head)",
        show_ylabel=True,
    )
    _plot_panel(
        axes[1],
        probs_cosine,
        langs,
        tgt_lang=tgt_lang,
        title="Cosine Lens Control (L2-normalized lm_head rows)",
        show_ylabel=False,
    )

    y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1], 0.05)
    axes[0].set_ylim(0.0, y_max)
    axes[1].set_ylim(0.0, y_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"{model_id} — {src_lang} → {tgt_lang} (n={n_words} concrete nouns, fp16)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_combined_grid(
    runs: list[tuple[str, str, Path]],
    lens_key: str,
    title: str,
    out_path: Path,
) -> Path | None:
    """Render a multi-model × 2-direction comparison grid for a given lens type."""
    if not runs:
        return None

    models_order: list[str] = []
    for m, _, _ in runs:
        if m not in models_order:
            models_order.append(m)

    directions = ["es-ca", "es-fr"]
    run_map = {(m, d): p for m, d, p in runs}

    n_rows = len(models_order)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12.5, max(3.6 * n_rows, 4.5)), dpi=180, squeeze=False
    )

    legend_handles = None
    legend_labels = None

    for r_idx, model_id in enumerate(models_order):
        short_name = model_id.split("/")[-1]
        for c_idx, direction in enumerate(directions):
            ax = axes[r_idx, c_idx]
            npz_path = run_map.get((model_id, direction))
            if npz_path is None or not npz_path.exists():
                ax.text(0.5, 0.5, "Not run", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{short_name} ({direction})")
                continue

            data = np.load(npz_path, allow_pickle=False)
            probs = data[lens_key]
            langs = [str(x) for x in data["langs"]]
            _, tgt_lang = parse_direction(direction)
            arrow = "es → ca" if direction == "es-ca" else "es → fr"
            _plot_panel(
                ax,
                probs,
                langs,
                tgt_lang=tgt_lang,
                title=f"{short_name} | {arrow} (n={probs.shape[0]})",
                show_ylabel=(c_idx == 0),
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
                # Remove '[target]' suffix in shared multi-direction legend
                legend_labels = [lbl.replace(" [target]", "") for lbl in legend_labels]

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_combined_figures(results_dir: str | Path = "results", figures_dir: str | Path = "figures") -> list[Path]:
    """Generate combined comparison figures from saved per-run .npz files."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        return []

    summary_df = pd.read_csv(summary_path)
    runs: list[tuple[str, str, Path]] = []
    for _, row in summary_df.iterrows():
        model_id = str(row["model"])
        direction = str(row["direction"])
        slug = model_id.split("/")[-1]
        npz_path = results_dir / f"probs_{slug}_{direction}.npz"
        if npz_path.exists():
            runs.append((model_id, direction, npz_path))

    created: list[Path] = []
    p1 = _plot_combined_grid(
        runs,
        lens_key="probs_logit",
        title="Cross-Model Comparison: Standard Logit Lens (es → ca vs es → fr)",
        out_path=figures_dir / "combined_comparison.png",
    )
    if p1 is not None:
        created.append(p1)

    p2 = _plot_combined_grid(
        runs,
        lens_key="probs_cosine",
        title="Cross-Model Comparison: Cosine Lens Control (L2-normalized lm_head)",
        out_path=figures_dir / "combined_cosine_comparison.png",
    )
    if p2 is not None:
        created.append(p2)

    return created

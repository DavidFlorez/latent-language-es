"""Logit lens and L2-normalized cosine lens implementation for HuggingFace CausalLMs."""

from __future__ import annotations

import gc
import os
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import ALL_LANGS, build_few_shot_prompt, normalize_direction, parse_direction


class GatedModelSkipError(RuntimeError):
    """Raised when a gated Hugging Face model cannot be accessed due to missing/invalid HF_TOKEN."""


def get_hf_token() -> str | None:
    """Resolve HF_TOKEN from environment variables, Colab userdata secrets, or HF cache."""
    for env_key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val

    try:
        from google.colab import userdata  # type: ignore

        val = userdata.get("HF_TOKEN")
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass

    try:
        from huggingface_hub import get_token

        val = get_token()
        if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
        pass

    return None


def _is_gated_or_auth_error(exc: Exception) -> bool:
    """Check whether an exception indicates missing/unauthorized HF authentication for a gated repo."""
    msg = str(exc).lower()
    keywords = (
        "gated",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "access to model",
        "you must have access",
        "authentication",
        "huggingface.co/settings/tokens",
    )
    return any(k in msg for k in keywords)


def load_tokenizer_safe(model_id: str, hf_token: str | None = None):
    """Load an HF tokenizer, raising GatedModelSkipError if gated and inaccessible."""
    try:
        return AutoTokenizer.from_pretrained(model_id, token=hf_token)
    except Exception as exc:
        if _is_gated_or_auth_error(exc):
            raise GatedModelSkipError(
                f"Model '{model_id}' is gated and HF_TOKEN is missing or unauthorized."
            ) from exc
        raise


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    hf_token: str | None = None,
):
    """Load HF CausalLM in fp16 on CUDA and its tokenizer, handling gated repos cleanly."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is False. "
            "Run this script on a Google Colab GPU runtime."
        )

    tokenizer = load_tokenizer_safe(model_id, hf_token=hf_token)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except Exception as exc:
        if _is_gated_or_auth_error(exc):
            raise GatedModelSkipError(
                f"Model '{model_id}' is gated and HF_TOKEN is missing or unauthorized."
            ) from exc
        raise

    model = model.to(device)
    model.eval()
    return model, tokenizer


def release_model(*objects: Any) -> None:
    """Delete model/tensor references and release CUDA memory."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def build_piece_to_token_ids(tokenizer, max_vocab_size: int | None = None) -> dict[str, list[int]]:
    """Decode every non-special vocabulary token preserving leading whitespace.

    Works identically across SentencePiece ('▁') and byte-level BPE ('Ġ') tokenizers.
    """
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    valid_ids = [
        int(tid)
        for tid in vocab.values()
        if tid not in special_ids and (max_vocab_size is None or int(tid) < max_vocab_size)
    ]
    valid_ids.sort()

    anchor_ids = tokenizer.encode("Hello", add_special_tokens=False)
    anchor_text = tokenizer.decode(anchor_ids, clean_up_tokenization_spaces=False)

    piece_to_ids: dict[str, list[int]] = {}
    chunk_size = 8192
    for start in range(0, len(valid_ids), chunk_size):
        chunk = valid_ids[start : start + chunk_size]
        seqs = [anchor_ids + [tid] for tid in chunk]
        decoded_list = tokenizer.batch_decode(
            seqs, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for tid, decoded in zip(chunk, decoded_list):
            if not decoded.startswith(anchor_text):
                continue
            piece = decoded[len(anchor_text) :]
            if not piece or not piece.strip() or "\ufffd" in piece:
                continue
            piece_nfc = unicodedata.normalize("NFC", piece)
            piece_to_ids.setdefault(piece_nfc, []).append(tid)

    return piece_to_ids


def get_prefix_token_ids(word: str, piece_to_ids: dict[str, list[int]]) -> list[int]:
    """Return all vocab token IDs that can begin `word` (with and without a leading space)."""
    w = unicodedata.normalize("NFC", word.strip())
    if not w:
        return []

    found_ids: set[int] = set()
    for i in range(1, len(w) + 1):
        prefix = w[:i]
        for candidate in (prefix, " " + prefix):
            if candidate in piece_to_ids:
                found_ids.update(piece_to_ids[candidate])
    return sorted(found_ids)


def check_tokenizers(
    df: pd.DataFrame,
    model_ids: list[str],
    hf_token: str | None = None,
) -> pd.DataFrame:
    """Inspect how many words per language are single-token for each model tokenizer."""
    records: list[dict[str, Any]] = []
    for model_id in model_ids:
        try:
            tokenizer = load_tokenizer_safe(model_id, hf_token=hf_token)
        except GatedModelSkipError as exc:
            print(f"[WARNING] Skipping tokenizer check for {model_id}: {exc}")
            continue

        vocab_size = len(tokenizer)
        for lang in ALL_LANGS:
            words = df[lang].tolist()
            lens_no_space = [len(tokenizer.encode(w, add_special_tokens=False)) for w in words]
            lens_with_space = [len(tokenizer.encode(" " + w, add_special_tokens=False)) for w in words]

            single_no_space = sum(1 for n in lens_no_space if n == 1)
            single_with_space = sum(1 for n in lens_with_space if n == 1)
            single_either = sum(
                1 for n0, n1 in zip(lens_no_space, lens_with_space) if (n0 == 1 or n1 == 1)
            )
            records.append(
                {
                    "model": model_id,
                    "vocab_size": vocab_size,
                    "lang": lang,
                    "total_words": len(words),
                    "single_token_no_space": single_no_space,
                    "single_token_with_space": single_with_space,
                    "single_token_either": single_either,
                    "pct_single_either": round(100.0 * single_either / len(words), 1),
                    "mean_tokens_no_space": round(float(np.mean(lens_no_space)), 2),
                    "mean_tokens_with_space": round(float(np.mean(lens_with_space)), 2),
                }
            )
        release_model(tokenizer)

    return pd.DataFrame(records)


def get_norm_and_lm_head(model):
    """Extract the final hidden-state norm module and output lm_head linear layer."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_norm = model.transformer.ln_f
    else:
        raise AttributeError(f"Could not locate final norm layer on model {type(model).__name__}")

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise AttributeError(f"Could not locate lm_head on model {type(model).__name__}")

    return final_norm, lm_head


@torch.inference_mode()
def detect_last_hidden_already_normed(model, final_norm, lm_head, tokenizer, device: str) -> bool:
    """Verify whether outputs.hidden_states[-1] already has final_norm applied."""
    test_inputs = tokenizer('Español: "perro" - Català: "', return_tensors="pt").to(device)
    outputs = model(**test_inputs, output_hidden_states=True, use_cache=False)
    last_hs = outputs.hidden_states[-1][:, -1, :]
    model_logits = outputs.logits[:, -1, :].float()

    logits_direct = lm_head(last_hs).float()
    logits_renorm = lm_head(final_norm(last_hs)).float()

    diff_direct = float((logits_direct - model_logits).abs().max().item())
    diff_renorm = float((logits_renorm - model_logits).abs().max().item())

    already_normed = diff_direct < diff_renorm and diff_direct < 0.05
    print(
        f"[Check] outputs.hidden_states[-1] vs model.logits max_abs_diff: "
        f"direct={diff_direct:.6f}, re-normed={diff_renorm:.6f} -> already_normed={already_normed}"
    )
    return already_normed


@dataclass
class LogitLensOutput:
    """Container for per-word, per-layer, per-language probabilities and example top-5 tokens."""

    model_id: str
    direction: str
    num_layers: int
    n_words: int
    langs: list[str]
    probs_logit: np.ndarray  # shape: (n_words, num_layers, len(langs))
    probs_cosine: np.ndarray  # shape: (n_words, num_layers, len(langs))
    top1_acc_logit: float
    top1_acc_cosine: float
    top5_records: list[dict[str, Any]]
    lm_head_norm_stats: dict[str, float]


@torch.inference_mode()
def run_logit_lens(
    model,
    tokenizer,
    model_id: str,
    df: pd.DataFrame,
    direction: str,
    device: str = "cuda",
    num_shots: int = 5,
    seed: int = 42,
    example_words: tuple[str, ...] = ("perro", "manzana", "queso"),
) -> LogitLensOutput:
    """Run standard logit lens and cosine lens across all nouns for one translation direction."""
    norm_dir = normalize_direction(direction)
    src_lang, tgt_lang = parse_direction(norm_dir)

    final_norm, lm_head = get_norm_and_lm_head(model)
    lm_head_vocab_size = lm_head.weight.shape[0]
    last_already_normed = detect_last_hidden_already_normed(
        model, final_norm, lm_head, tokenizer, device
    )

    piece_to_ids = build_piece_to_token_ids(tokenizer, max_vocab_size=lm_head_vocab_size)
    id_to_piece = {tid: piece for piece, tids in piece_to_ids.items() for tid in tids}

    # Precompute L2-normalized unembedding weights once for cosine lens control.
    w_float = lm_head.weight.detach().float()
    row_norms = w_float.norm(p=2, dim=-1)
    w_cos = F.normalize(w_float, p=2, dim=-1)
    del w_float

    n_words = len(df)
    num_layers = int(model.config.num_hidden_layers)

    probs_logit = np.zeros((n_words, num_layers, len(ALL_LANGS)), dtype=np.float32)
    probs_cosine = np.zeros((n_words, num_layers, len(ALL_LANGS)), dtype=np.float32)

    top1_hits_logit = 0
    top1_hits_cosine = 0
    top5_records: list[dict[str, Any]] = []
    lang_row_norms: dict[str, list[float]] = {lang: [] for lang in ALL_LANGS}

    example_set = set(example_words)
    anchor_ids = tokenizer.encode("Hello", add_special_tokens=False)
    anchor_text = tokenizer.decode(anchor_ids, clean_up_tokenization_spaces=False)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    def _format_tid(tid: int) -> str:
        if tid in id_to_piece:
            return id_to_piece[tid]
        if tid in special_ids:
            return tokenizer.decode([tid], skip_special_tokens=False)
        decoded = tokenizer.decode(anchor_ids + [tid], clean_up_tokenization_spaces=False)
        if decoded.startswith(anchor_text):
            return decoded[len(anchor_text) :]
        return tokenizer.decode([tid], clean_up_tokenization_spaces=False)

    for idx in tqdm(range(n_words), desc=f"{model_id.split('/')[-1]} [{norm_dir}]"):
        row = df.iloc[idx]
        prompt = build_few_shot_prompt(df, idx, norm_dir, num_shots=num_shots, seed=seed)

        lang_token_ids: dict[str, list[int]] = {}
        lang_tensors: dict[str, torch.Tensor] = {}
        for lang in ALL_LANGS:
            tids = get_prefix_token_ids(row[lang], piece_to_ids)
            lang_token_ids[lang] = tids
            if tids:
                t = torch.tensor(tids, dtype=torch.long, device=device)
                lang_tensors[lang] = t
                lang_row_norms[lang].append(float(row_norms[t].mean().item()))

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # hidden_states[1:] corresponds to transformer layers 1 .. num_layers.
        layer_states = outputs.hidden_states[1:]
        assert len(layer_states) == num_layers, (
            f"Expected {num_layers} hidden states, got {len(layer_states)}"
        )

        # Collect last prompt token hidden state per layer and apply final_norm appropriately.
        normed_layers = []
        for l_idx, hs in enumerate(layer_states):
            last_h = hs[:, -1, :]
            if l_idx == num_layers - 1 and last_already_normed:
                normed_layers.append(last_h.squeeze(0))
            else:
                normed_layers.append(final_norm(last_h).squeeze(0))

        normed_stack = torch.stack(normed_layers, dim=0)  # (L, d)

        logits_std = lm_head(normed_stack).float()  # (L, V)
        logits_cos = torch.matmul(normed_stack.float(), w_cos.T)  # (L, V)

        p_std = torch.softmax(logits_std, dim=-1)
        p_cos = torch.softmax(logits_cos, dim=-1)

        for lang_i, lang in enumerate(ALL_LANGS):
            if lang in lang_tensors:
                t_idx = lang_tensors[lang]
                probs_logit[idx, :, lang_i] = p_std[:, t_idx].sum(dim=-1).cpu().numpy()
                probs_cosine[idx, :, lang_i] = p_cos[:, t_idx].sum(dim=-1).cpu().numpy()

        tgt_set = set(lang_token_ids[tgt_lang])
        if int(p_std[-1].argmax().item()) in tgt_set:
            top1_hits_logit += 1
        if int(p_cos[-1].argmax().item()) in tgt_set:
            top1_hits_cosine += 1

        if row["es"] in example_set:
            for lens_name, p_mat in (("logit_lens", p_std), ("cosine_lens", p_cos)):
                top_vals, top_ids = torch.topk(p_mat, k=5, dim=-1)
                top_vals_cpu = top_vals.cpu().numpy()
                top_ids_cpu = top_ids.cpu().numpy()
                for l_idx in range(num_layers):
                    tokens_formatted = []
                    for rank_i in range(5):
                        tid = int(top_ids_cpu[l_idx, rank_i])
                        prob = float(top_vals_cpu[l_idx, rank_i])
                        piece = _format_tid(tid)
                        tokens_formatted.append((piece, prob))

                    rec: dict[str, Any] = {
                        "model": model_id,
                        "direction": norm_dir,
                        "lens": lens_name,
                        "word_es": row["es"],
                        "word_target": row[tgt_lang],
                        "word_en": row["en"],
                        "layer": l_idx + 1,
                    }
                    for rank_i, (piece, prob) in enumerate(tokens_formatted, start=1):
                        rec[f"top{rank_i}_token"] = repr(piece)
                        rec[f"top{rank_i}_prob"] = round(prob, 4)
                    top5_records.append(rec)

        del inputs, outputs, layer_states, normed_layers, normed_stack
        del logits_std, logits_cos, p_std, p_cos

    del w_cos, row_norms
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lm_head_norm_stats = {
        lang: round(float(np.mean(vals)), 4) if vals else float("nan")
        for lang, vals in lang_row_norms.items()
    }

    return LogitLensOutput(
        model_id=model_id,
        direction=norm_dir,
        num_layers=num_layers,
        n_words=n_words,
        langs=list(ALL_LANGS),
        probs_logit=probs_logit,
        probs_cosine=probs_cosine,
        top1_acc_logit=top1_hits_logit / n_words,
        top1_acc_cosine=top1_hits_cosine / n_words,
        top5_records=top5_records,
        lm_head_norm_stats=lm_head_norm_stats,
    )

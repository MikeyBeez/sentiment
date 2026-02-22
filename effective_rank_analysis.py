"""
Effective rank analysis of hidden state representations.

For each trained model (3 models × 2 datasets), loads the final epoch checkpoint,
runs the validation set through the backbone, collects the mean-pooled hidden states,
and computes:
  - Singular value spectrum
  - Effective rank: exp(entropy of normalized singular values)
  - Participation ratio: (sum(sigma))^2 / sum(sigma^2)
  - Top-k variance explained (how many dimensions capture 90%, 95%, 99% of variance)

If the anti-collapse hypothesis is correct, Models 2 and 3 should have higher
effective rank than Model 1, and the difference should be larger on AG News
than on SST-2.
"""

import os, gc, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-7B"
DEVICE = "cuda"
SEED = 42
MAX_SAMPLES = 2000  # subsample validation sets for tractable SVD

torch.manual_seed(SEED)
np.random.seed(SEED)

DATASETS = {
    "sst2": {
        "max_length": 128,
        "results_dir": "/home/bee/Code/sentiment/results_7b",
        "num_classes": 2,
    },
    "agnews": {
        "max_length": 256,
        "results_dir": "/home/bee/Code/sentiment/results_agnews",
        "num_classes": 4,
    },
}

OUT_DIR = "/home/bee/Code/sentiment/results_effective_rank"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data loading ─────────────────────────────────────────────────────────────

def load_val_data(dataset_name, tokenizer):
    if dataset_name == "sst2":
        raw = load_dataset("glue", "sst2")
        val = raw["validation"]
        max_len = DATASETS["sst2"]["max_length"]
        def tok_fn(examples):
            return tokenizer(examples["sentence"], padding="max_length",
                             truncation=True, max_length=max_len)
    else:
        raw = load_dataset("ag_news")
        val = raw["test"]
        max_len = DATASETS["agnews"]["max_length"]
        def tok_fn(examples):
            return tokenizer(examples["text"], padding="max_length",
                             truncation=True, max_length=max_len)

    val = val.map(tok_fn, batched=True)
    val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Subsample for tractable SVD
    if len(val) > MAX_SAMPLES:
        indices = np.random.choice(len(val), MAX_SAMPLES, replace=False)
        val = val.select(indices.tolist())

    return val

# ── Model loading ────────────────────────────────────────────────────────────

def load_checkpoint(dataset_name, model_type, epoch=3):
    """Load a trained model from checkpoint."""
    results_dir = DATASETS[dataset_name]["results_dir"]
    ckpt_dir = os.path.join(results_dir, f"model{model_type}", f"epoch{epoch}")
    lora_dir = os.path.join(ckpt_dir, "lora")

    print(f"  Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"  Loading LoRA from {lora_dir}...")
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    return model

# ── Hidden state collection ──────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_states(model, val_dl):
    """Run validation data through the model and collect mean-pooled hidden states."""
    all_hidden = []
    all_labels = []

    backbone = model.get_base_model().model  # the transformer body (not lm_head)

    for batch in tqdm(val_dl, desc="    Collecting hidden states", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"]

        out = backbone(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state  # (B, L, H)

        # Mean pool over non-padding tokens
        pool_mask = mask.unsqueeze(-1).float()
        pooled = (hidden.float() * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)

        all_hidden.append(pooled.cpu())
        all_labels.append(labels)

    hidden_matrix = torch.cat(all_hidden, dim=0).numpy()  # (N, H)
    labels = torch.cat(all_labels, dim=0).numpy()
    return hidden_matrix, labels

# ── Effective rank metrics ───────────────────────────────────────────────────

def compute_rank_metrics(hidden_matrix):
    """Compute effective rank and related metrics from an (N, D) matrix."""
    # Center the data
    centered = hidden_matrix - hidden_matrix.mean(axis=0, keepdims=True)

    # SVD
    U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)

    # Normalized singular values (as distribution)
    sigma_pos = sigma[sigma > 1e-10]
    p = sigma_pos / sigma_pos.sum()

    # Effective rank: exp(entropy of normalized singular values)
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)

    # Participation ratio: (sum(sigma))^2 / sum(sigma^2)
    participation_ratio = (sigma_pos.sum() ** 2) / (sigma_pos ** 2).sum()

    # Variance explained by top-k
    variance = sigma ** 2
    total_var = variance.sum()
    cumvar = np.cumsum(variance) / total_var

    dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    dims_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    # Stable rank: ||A||_F^2 / ||A||_2^2 = sum(sigma^2) / sigma_max^2
    stable_rank = (sigma ** 2).sum() / (sigma[0] ** 2) if sigma[0] > 0 else 0

    return {
        "effective_rank": round(float(effective_rank), 2),
        "participation_ratio": round(float(participation_ratio), 2),
        "stable_rank": round(float(stable_rank), 2),
        "dims_90pct": int(dims_90),
        "dims_95pct": int(dims_95),
        "dims_99pct": int(dims_99),
        "top1_variance": round(float(cumvar[0]), 4),
        "top10_variance": round(float(cumvar[min(9, len(cumvar)-1)]), 4),
        "top50_variance": round(float(cumvar[min(49, len(cumvar)-1)]), 4),
        "num_dimensions": len(sigma),
        "sigma": sigma,  # keep for plotting
        "cumvar": cumvar,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Also measure pretrained baseline (no fine-tuning)
    all_results = {}

    for ds_name in ["sst2", "agnews"]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'='*60}")

        val_ds = load_val_data(ds_name, tokenizer)
        print(f"  Validation samples: {len(val_ds)}")
        val_dl = DataLoader(val_ds, batch_size=8, shuffle=False)

        ds_results = {}

        # Pretrained baseline (no LoRA, no fine-tuning)
        print(f"\n  --- Pretrained baseline (no fine-tuning) ---")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model.eval()

        # Collect hidden states from pretrained model directly
        all_hidden_pre = []
        all_labels_pre = []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="    Pretrained hidden states", leave=False):
                ids = batch["input_ids"].to(DEVICE)
                mask = batch["attention_mask"].to(DEVICE)
                out = base_model.model(input_ids=ids, attention_mask=mask)
                hidden = out.last_hidden_state
                pool_mask = mask.unsqueeze(-1).float()
                pooled = (hidden.float() * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
                all_hidden_pre.append(pooled.cpu())
                all_labels_pre.append(batch["label"])

        hidden_pre = torch.cat(all_hidden_pre).numpy()
        metrics_pre = compute_rank_metrics(hidden_pre)
        ds_results["pretrained"] = {k: v for k, v in metrics_pre.items()
                                     if k not in ("sigma", "cumvar")}
        ds_results["pretrained"]["_sigma"] = metrics_pre["sigma"]
        ds_results["pretrained"]["_cumvar"] = metrics_pre["cumvar"]
        print(f"    Effective rank: {metrics_pre['effective_rank']}")
        print(f"    Stable rank:    {metrics_pre['stable_rank']}")
        print(f"    Dims for 95%:   {metrics_pre['dims_95pct']}")

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        # Fine-tuned models
        for model_type in [1, 2, 3]:
            model_names = {1: "Cls Only", 2: "Dual Head", 3: "Dual + Engram"}
            print(f"\n  --- Model {model_type}: {model_names[model_type]} ---")

            model = load_checkpoint(ds_name, model_type, epoch=3)
            hidden_matrix, labels = collect_hidden_states(model, val_dl)

            metrics = compute_rank_metrics(hidden_matrix)
            key = f"model{model_type}"
            ds_results[key] = {k: v for k, v in metrics.items()
                               if k not in ("sigma", "cumvar")}
            ds_results[key]["_sigma"] = metrics["sigma"]
            ds_results[key]["_cumvar"] = metrics["cumvar"]

            print(f"    Effective rank: {metrics['effective_rank']}")
            print(f"    Stable rank:    {metrics['stable_rank']}")
            print(f"    Part. ratio:    {metrics['participation_ratio']}")
            print(f"    Dims for 90%:   {metrics['dims_90pct']}")
            print(f"    Dims for 95%:   {metrics['dims_95pct']}")
            print(f"    Dims for 99%:   {metrics['dims_99pct']}")
            print(f"    Top-1 var:      {metrics['top1_variance']}")
            print(f"    Top-10 var:     {metrics['top10_variance']}")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        all_results[ds_name] = ds_results

    # ── Visualization ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, ds_name in enumerate(["sst2", "agnews"]):
        ds = all_results[ds_name]
        ds_label = "SST-2" if ds_name == "sst2" else "AG News"

        # Plot 1: Singular value spectrum (log scale)
        ax = axes[row, 0]
        for key, label, color in [("pretrained", "Pretrained", "#95a5a6"),
                                   ("model1", "M1: Cls Only", "#e74c3c"),
                                   ("model2", "M2: Dual Head", "#3498db"),
                                   ("model3", "M3: Dual+Engram", "#2ecc71")]:
            sigma = ds[key]["_sigma"]
            ax.plot(range(len(sigma)), sigma, label=label, color=color, alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("Singular value index")
        ax.set_ylabel("Singular value (log)")
        ax.set_title(f"{ds_label} — Singular Value Spectrum")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 2: Cumulative variance explained
        ax = axes[row, 1]
        for key, label, color in [("pretrained", "Pretrained", "#95a5a6"),
                                   ("model1", "M1: Cls Only", "#e74c3c"),
                                   ("model2", "M2: Dual Head", "#3498db"),
                                   ("model3", "M3: Dual+Engram", "#2ecc71")]:
            cumvar = ds[key]["_cumvar"]
            ax.plot(range(len(cumvar)), cumvar, label=label, color=color, alpha=0.8)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
        ax.set_xlabel("Number of dimensions")
        ax.set_ylabel("Cumulative variance explained")
        ax.set_title(f"{ds_label} — Cumulative Variance")
        ax.set_xlim(0, 200)  # zoom into first 200 dims
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot 3: Effective rank bar chart
        ax = axes[row, 2]
        models = ["Pretrained", "M1: Cls Only", "M2: Dual", "M3: Dual+Eng"]
        keys = ["pretrained", "model1", "model2", "model3"]
        colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71"]
        ranks = [ds[k]["effective_rank"] for k in keys]
        bars = ax.bar(models, ranks, color=colors, alpha=0.8)
        for bar, rank in zip(bars, ranks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{rank:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Effective Rank")
        ax.set_title(f"{ds_label} — Effective Rank")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Representational Collapse Analysis: Effective Rank of Hidden States",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "effective_rank.png"), dpi=150, bbox_inches="tight")
    print(f"\n  Saved effective_rank.png")

    # ── Summary table ────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"  EFFECTIVE RANK ANALYSIS — SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Eff. Rank':>10} {'Stable Rank':>12} {'Part. Ratio':>12} {'Dims@95%':>10}")
    print(f"  {'-'*69}")

    for ds_name in ["sst2", "agnews"]:
        ds = all_results[ds_name]
        ds_label = "SST-2" if ds_name == "sst2" else "AG News"
        print(f"\n  {ds_label}:")
        for key, label in [("pretrained", "Pretrained (no FT)"),
                           ("model1", "Model 1: Cls Only"),
                           ("model2", "Model 2: Dual Head"),
                           ("model3", "Model 3: Dual+Engram")]:
            m = ds[key]
            print(f"    {label:<23} {m['effective_rank']:>10.1f} "
                  f"{m['stable_rank']:>12.1f} {m['participation_ratio']:>12.1f} "
                  f"{m['dims_95pct']:>10d}")

    # Anti-collapse test
    print(f"\n  {'='*70}")
    print(f"  ANTI-COLLAPSE HYPOTHESIS TEST")
    print(f"  {'='*70}")
    for ds_name in ["sst2", "agnews"]:
        ds = all_results[ds_name]
        ds_label = "SST-2" if ds_name == "sst2" else "AG News"
        pre = ds["pretrained"]["effective_rank"]
        m1 = ds["model1"]["effective_rank"]
        m2 = ds["model2"]["effective_rank"]
        m3 = ds["model3"]["effective_rank"]
        print(f"\n  {ds_label}:")
        print(f"    Pretrained → Cls Only:   {m1 - pre:>+8.1f}  (collapse from fine-tuning)")
        print(f"    Cls Only → Dual Head:    {m2 - m1:>+8.1f}  (recovery from dual objective)")
        print(f"    Dual Head → Dual+Engram: {m3 - m2:>+8.1f}  (engram effect)")
        print(f"    Collapse prevented:      {((m2 - m1) / max(abs(pre - m1), 0.01)) * 100:>7.1f}%  "
              f"(dual obj recovers this fraction of collapse)")

    # Save results (without numpy arrays)
    save_results = {}
    for ds_name in all_results:
        save_results[ds_name] = {}
        for key in all_results[ds_name]:
            save_results[ds_name][key] = {
                k: v for k, v in all_results[ds_name][key].items()
                if not k.startswith("_")
            }

    with open(os.path.join(OUT_DIR, "effective_rank_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Saved effective_rank_results.json")


if __name__ == "__main__":
    main()

"""
7B-scale dual-head topic classification experiment with engram feedback.

AG News generalization test: same architecture as the SST-2 experiment
(experiment_7b.py) applied to 4-class topic classification to test whether
the engram bottleneck mechanism generalizes beyond sentiment.

Three models trained on AG News with QLoRA on Qwen2.5-7B:
  1. Classification-only (LoRA backbone + topic head, no LM loss)
  2. Dual-head (LoRA backbone + topic head + LM loss)
  3. Dual-head + engram (same as 2, with 128-dim bottleneck fed back as prefix)

If the mechanism is general, engram clusters should align with topic categories
(World, Sports, Business, Sci/Tech) rather than sentiment polarity.
"""

import os, gc, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME    = "Qwen/Qwen2.5-7B"
MAX_LENGTH    = 256        # AG News texts are longer than SST-2
BATCH_SIZE    = 8          # per-device batch; Model 3 uses 4 (paired halves)
GRAD_ACCUM    = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS        = 3
LR            = 2e-4
WARMUP_RATIO  = 0.05
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
ENGRAM_DIM    = 128
NUM_CLASSES   = 4          # World, Sports, Business, Sci/Tech
SEED          = 42
DEVICE        = "cuda"
OUT_DIR       = "/home/bee/Code/sentiment/results_agnews"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(tokenizer):
    raw = load_dataset("ag_news")

    def tok_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train = raw["train"].map(tok_fn, batched=True)
    val   = raw["test"].map(tok_fn, batched=True)   # AG News test split as validation
    train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train, val

# ── Model wrapper ────────────────────────────────────────────────────────────

class DualHeadWrapper(nn.Module):
    """Wraps a QLoRA CausalLM with classification head and optional engram."""

    def __init__(self, peft_model, hidden_size, model_type, pad_token_id):
        super().__init__()
        self.peft_model = peft_model
        self.causal_lm  = peft_model.get_base_model()
        self.model_type  = model_type
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # Classification head — split into two stages for engram extraction
        # Stage 1: project to intermediate (engram taps here)
        self.cls_first = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        ).float()
        # Stage 2: final classification
        self.cls_final = nn.Linear(256, NUM_CLASSES).float()

        # Engram components — compress from cls intermediate (256→128)
        self.use_engram = (model_type == 3)
        if self.use_engram:
            self.engram_compressor = nn.Sequential(
                nn.Linear(256, ENGRAM_DIM),
                nn.Tanh(),
            ).float()
            self.engram_projector = nn.Linear(ENGRAM_DIM, hidden_size).float()

    def forward(self, input_ids, attention_mask, labels=None, engram=None):
        B = input_ids.shape[0]
        orig_mask = attention_mask

        # Get input embeddings
        embeds = self.causal_lm.model.embed_tokens(input_ids)

        # Prepend engram as a learned prefix token
        has_prefix = self.use_engram and engram is not None
        if has_prefix:
            prefix = self.engram_projector(engram).unsqueeze(1).to(embeds.dtype)
            embeds = torch.cat([prefix, embeds], dim=1)
            pfx = torch.ones(B, 1, device=DEVICE, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pfx, attention_mask], dim=1)

        # Transformer forward
        out = self.causal_lm.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state          # (B, L[+1], H)

        # ── LM loss (models 2 and 3) ────────────────────────────────────────
        lm_loss = None
        if self.model_type in (2, 3):
            lm_logits = self.causal_lm.lm_head(hidden)
            shift_logits = lm_logits[:, :-1, :].contiguous()

            lm_labels = input_ids.clone()
            lm_labels[orig_mask == 0] = -100          # mask padding
            if has_prefix:
                ignore = torch.full((B, 1), -100, device=DEVICE, dtype=torch.long)
                lm_labels = torch.cat([ignore, lm_labels], dim=1)

            shift_labels = lm_labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        # ── Mean pool for classification / engram ────────────────────────────
        token_hidden = hidden[:, 1:, :] if has_prefix else hidden
        pool_mask = orig_mask.unsqueeze(-1).float()
        pooled = (token_hidden.float() * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)

        # ── Classification head (fp32) ───────────────────────────────────────
        cls_intermediate = self.cls_first(pooled)       # (B, 256)
        cls_logits = self.cls_final(cls_intermediate)   # (B, NUM_CLASSES)
        cls_loss = F.cross_entropy(cls_logits, labels) if labels is not None else None

        # ── Engram compressor — taps cls intermediate (fp32) ─────────────────
        new_engram = None
        if self.use_engram:
            new_engram = self.engram_compressor(cls_intermediate)

        # ── Combined loss ────────────────────────────────────────────────────
        if self.model_type == 1:
            total_loss = cls_loss
        else:
            total_loss = 0.5 * lm_loss + 0.5 * cls_loss

        return {
            "loss": total_loss,
            "cls_logits": cls_logits,
            "cls_loss": cls_loss,
            "lm_loss": lm_loss,
            "engram": new_engram,
        }

# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_type, pad_token_id):
    print(f"  Loading {MODEL_NAME} in 4-bit...")
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
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_config)

    hidden_size = base.config.hidden_size
    wrapper = DualHeadWrapper(peft_model, hidden_size, model_type, pad_token_id)
    wrapper.cls_first.to(DEVICE)
    wrapper.cls_final.to(DEVICE)
    if model_type == 3:
        wrapper.engram_compressor.to(DEVICE)
        wrapper.engram_projector.to(DEVICE)

    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in wrapper.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return wrapper

# ── Checkpoint saving ────────────────────────────────────────────────────────

def save_checkpoint(wrapper, epoch, model_type, metrics):
    ckpt_dir = os.path.join(OUT_DIR, f"model{model_type}", f"epoch{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    wrapper.peft_model.save_pretrained(os.path.join(ckpt_dir, "lora"))
    heads = {
        "cls_first": wrapper.cls_first.state_dict(),
        "cls_final": wrapper.cls_final.state_dict(),
    }
    if model_type == 3:
        heads["engram_compressor"] = wrapper.engram_compressor.state_dict()
        heads["engram_projector"]  = wrapper.engram_projector.state_dict()
    torch.save(heads, os.path.join(ckpt_dir, "heads.pt"))
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model_type, train_ds, val_ds, tokenizer):
    model_names = {1: "Classification Only", 2: "Dual Head", 3: "Dual + Engram"}
    name = model_names[model_type]
    print(f"\n{'='*60}")
    print(f"  Model {model_type}: {name}")
    print(f"{'='*60}")

    wrapper = load_model(model_type, tokenizer.pad_token_id)

    # DataLoader — Model 3 uses batch of 4 (split into pairs of 2)
    bs = 4 if model_type == 3 else BATCH_SIZE
    ga = GRAD_ACCUM * 2 if model_type == 3 else GRAD_ACCUM  # keep effective batch ~same
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=bs, shuffle=False)

    # Optimizer — only trainable params
    trainable_params = [p for p in wrapper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    # Cosine schedule over full run
    steps_per_epoch = math.ceil(len(train_dl) / ga)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    all_metrics = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train epoch ──────────────────────────────────────────────────────
        wrapper.train()
        epoch_loss, epoch_cls, epoch_lm = 0.0, 0.0, 0.0
        epoch_correct, epoch_total = 0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_dl, desc=f"  Epoch {epoch}/{EPOCHS}", leave=False)
        for step, batch in enumerate(pbar):
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            B = ids.shape[0]

            if model_type == 3:
                # Sequential pairs: first half → engram → second half
                half = B // 2
                if half == 0:
                    continue

                zero_eng = torch.zeros(half, ENGRAM_DIM, device=DEVICE)
                out1 = wrapper(ids[:half], mask[:half], labels[:half], zero_eng)
                engram_for_second = out1["engram"]  # NOT detached

                out2 = wrapper(ids[half:2*half], mask[half:2*half],
                               labels[half:2*half], engram_for_second)

                loss = (out1["loss"] + out2["loss"]) / (2.0 * ga)
                loss.backward()

                # Metrics
                with torch.no_grad():
                    sl = out1["cls_loss"].item() + out2["cls_loss"].item()
                    epoch_cls += sl / 2
                    if out1["lm_loss"] is not None:
                        epoch_lm += (out1["lm_loss"].item() + out2["lm_loss"].item()) / 2
                    p1 = out1["cls_logits"].argmax(-1)
                    p2 = out2["cls_logits"].argmax(-1)
                    epoch_correct += (p1 == labels[:half]).sum().item()
                    epoch_correct += (p2 == labels[half:2*half]).sum().item()
                    epoch_total += 2 * half
                    epoch_loss += (out1["loss"].item() + out2["loss"].item()) / 2
            else:
                out = wrapper(ids, mask, labels)
                loss = out["loss"] / ga
                loss.backward()

                with torch.no_grad():
                    epoch_cls += out["cls_loss"].item()
                    if out["lm_loss"] is not None:
                        epoch_lm += out["lm_loss"].item()
                    preds = out["cls_logits"].argmax(-1)
                    epoch_correct += (preds == labels).sum().item()
                    epoch_total += B
                    epoch_loss += out["loss"].item()

            if (step + 1) % ga == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}",
                                 acc=f"{epoch_correct/max(epoch_total,1):.3f}")

        # Flush remaining gradients
        if (step + 1) % ga != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        n = len(train_dl)
        train_metrics = {
            "epoch": epoch,
            "loss": round(epoch_loss / n, 4),
            "cls_loss": round(epoch_cls / n, 4),
            "lm_loss": round(epoch_lm / n, 4) if model_type != 1 else None,
            "accuracy": round(epoch_correct / max(epoch_total, 1), 4),
        }
        all_metrics["train"].append(train_metrics)

        # ── Validate ─────────────────────────────────────────────────────────
        val_metrics = evaluate(wrapper, val_dl, model_type)
        val_metrics["epoch"] = epoch
        all_metrics["val"].append(val_metrics)

        print(f"  Epoch {epoch}: train_acc={train_metrics['accuracy']:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_cls_loss={val_metrics['cls_loss']:.4f}"
              + (f"  val_lm_loss={val_metrics['lm_loss']:.4f}" if val_metrics['lm_loss'] else ""))

        save_checkpoint(wrapper, epoch, model_type, all_metrics)

    return wrapper, all_metrics, val_dl

# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(wrapper, val_dl, model_type, collect_engrams=False):
    wrapper.eval()
    correct, total = 0, 0
    total_cls, total_lm = 0.0, 0.0
    all_engrams, all_labels = [], []
    all_engram_tensors, all_label_tensors = [], []

    for batch in val_dl:
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        B = ids.shape[0]

        if model_type == 3:
            eng = torch.zeros(B, ENGRAM_DIM, device=DEVICE)
            out = wrapper(ids, mask, labels, eng)
        else:
            out = wrapper(ids, mask, labels)

        preds = out["cls_logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total += B
        total_cls += out["cls_loss"].item()
        if out["lm_loss"] is not None:
            total_lm += out["lm_loss"].item()

        if collect_engrams and out["engram"] is not None:
            all_engrams.append(out["engram"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_engram_tensors.append(out["engram"].cpu())
            all_label_tensors.append(labels.cpu())

    n = len(val_dl)
    metrics = {
        "accuracy": round(correct / max(total, 1), 4),
        "cls_loss": round(total_cls / n, 4),
        "lm_loss": round(total_lm / n, 4) if model_type != 1 else None,
    }

    if collect_engrams and all_engrams:
        engrams_np = np.concatenate(all_engrams)
        labels_np  = np.concatenate(all_labels)
        eng_t = torch.cat(all_engram_tensors)
        lab_t = torch.cat(all_label_tensors)

        # Cosine similarity: within-class vs cross-class
        normed = F.normalize(eng_t, dim=1)
        # Compute in chunks to avoid OOM on 7600x7600 matrix
        n_samples = len(eng_t)
        chunk_size = 1000
        within_sum, within_count = 0.0, 0
        between_sum, between_count = 0.0, 0
        for i in range(0, n_samples, chunk_size):
            chunk_normed = normed[i:i+chunk_size]
            chunk_labels = lab_t[i:i+chunk_size]
            cos_chunk = chunk_normed @ normed.T  # (chunk, N)
            same = chunk_labels.unsqueeze(1) == lab_t.unsqueeze(0)
            # Exclude self-similarity on the diagonal block
            eye_block = torch.zeros_like(same)
            for j in range(len(chunk_labels)):
                global_idx = i + j
                if global_idx < n_samples:
                    eye_block[j, global_idx] = True
            same_no_diag = same & ~eye_block
            diff = ~same
            within_sum += cos_chunk[same_no_diag].sum().item()
            within_count += same_no_diag.sum().item()
            between_sum += cos_chunk[diff].sum().item()
            between_count += diff.sum().item()

        metrics["engram_cos_within"]  = round(within_sum / max(within_count, 1), 4)
        metrics["engram_cos_between"] = round(between_sum / max(between_count, 1), 4)
        return metrics, engrams_np, labels_np

    return metrics

# ── Clustering analysis ──────────────────────────────────────────────────────

def cluster_accuracy_hungarian(true_labels, cluster_labels, n_classes):
    """Compute cluster accuracy using Hungarian algorithm for optimal mapping."""
    # Build cost matrix: rows=clusters, cols=true classes
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for c, t in zip(cluster_labels, true_labels):
        if c < n_classes and t < n_classes:
            cost[c, t] += 1
    # Hungarian wants to minimize, so negate for maximum matching
    row_ind, col_ind = linear_sum_assignment(-cost)
    # Build mapping
    mapping = dict(zip(row_ind, col_ind))
    mapped = np.array([mapping.get(c, -1) for c in cluster_labels])
    return accuracy_score(true_labels, mapped), mapping

def cluster_analysis(engrams, labels):
    print(f"\n{'='*60}")
    print(f"  ENGRAM CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    print(f"  Engram shape: {engrams.shape}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(engrams)-1))
    proj = tsne.fit_transform(engrams)

    km = KMeans(n_clusters=NUM_CLASSES, random_state=SEED, n_init=10)
    clusters = km.fit_predict(engrams)

    ari = adjusted_rand_score(labels, clusters)
    cluster_acc, mapping = cluster_accuracy_hungarian(labels, clusters, NUM_CLASSES)

    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Cluster accuracy (Hungarian): {cluster_acc:.4f}")
    print(f"  Cluster→Class mapping: {{{', '.join(f'{k}→{CLASS_NAMES[v]}' for k, v in sorted(mapping.items()))}}}")

    if ari > 0.5:
        print("  -> Strong topic structure in engram space!")
    elif ari > 0.2:
        print("  -> Moderate topic structure in engram space.")
    elif ari > 0.05:
        print("  -> Weak topic structure in engram space.")
    else:
        print("  -> No clear topic separation.")

    # Per-class cluster purity
    print("\n  Per-class breakdown:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_mask = labels == cls_idx
        cls_clusters = clusters[cls_mask]
        dominant = np.bincount(cls_clusters, minlength=NUM_CLASSES).max()
        purity = dominant / cls_mask.sum() if cls_mask.sum() > 0 else 0
        print(f"    {cls_name}: n={cls_mask.sum()}, purity={purity:.3f}")

    # ── Visualization ────────────────────────────────────────────────────────
    topic_colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]  # World, Sports, Business, Sci/Tech
    cluster_colors_map = ["#e67e22", "#1abc9c", "#f39c12", "#8e44ad"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: true labels
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        ax1.scatter(proj[mask, 0], proj[mask, 1],
                    c=topic_colors[cls_idx], alpha=0.4, s=8, label=cls_name)
    ax1.set_title("Engram t-SNE — TRUE topic labels")
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")
    ax1.legend(markerscale=3, fontsize=9)

    # Right: K-Means clusters
    for k in range(NUM_CLASSES):
        mask = clusters == k
        mapped_name = CLASS_NAMES[mapping[k]] if k in mapping else f"Cluster {k}"
        ax2.scatter(proj[mask, 0], proj[mask, 1],
                    c=cluster_colors_map[k], alpha=0.4, s=8,
                    label=f"K{k}→{mapped_name}")
    ax2.set_title(f"Engram t-SNE — K-Means (k={NUM_CLASSES})\nARI={ari:.3f}, Acc={cluster_acc:.3f}")
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")
    ax2.legend(markerscale=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "engram_clusters.png"), dpi=150)
    print(f"\n  Saved engram_clusters.png")

    return {
        "ari": round(ari, 4),
        "cluster_accuracy": round(cluster_acc, 4),
        "mapping": {str(k): CLASS_NAMES[v] for k, v in mapping.items()},
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer (shared across all models)
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Data
    print("Loading AG News...")
    train_ds, val_ds = load_data(tokenizer)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    results = {}

    for model_type in [1, 2, 3]:
        t0 = time.time()
        wrapper, metrics, val_dl = train_model(model_type, train_ds, val_ds, tokenizer)

        # Collect engrams for Model 3
        if model_type == 3:
            val_final, engrams, elabels = evaluate(
                wrapper, val_dl, model_type, collect_engrams=True
            )
            cluster_res = cluster_analysis(engrams, elabels)
            metrics["cluster"] = cluster_res
            metrics["engram_cos"] = {
                "within": val_final.get("engram_cos_within"),
                "between": val_final.get("engram_cos_between"),
            }

        elapsed = time.time() - t0
        metrics["training_time_min"] = round(elapsed / 60, 1)
        results[f"model{model_type}"] = metrics

        # Free GPU memory before next model
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Training time: {elapsed/60:.1f} min")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — AG News")
    print(f"{'='*60}")

    for mt in [1, 2, 3]:
        key = f"model{mt}"
        val = results[key]["val"][-1]
        print(f"  Model {mt}: val_acc={val['accuracy']:.4f}  "
              f"cls_loss={val['cls_loss']:.4f}"
              + (f"  lm_loss={val['lm_loss']:.4f}" if val.get("lm_loss") else ""))

    m1 = results["model1"]["val"][-1]["accuracy"]
    m2 = results["model2"]["val"][-1]["accuracy"]
    m3 = results["model3"]["val"][-1]["accuracy"]

    print(f"\n  Model 2 vs 1 (dual-objective regularization): {m2-m1:+.4f}")
    print(f"  Model 3 vs 2 (engram feedback):               {m3-m2:+.4f}")

    if "cluster" in results["model3"]:
        c = results["model3"]["cluster"]
        print(f"  Engram ARI: {c['ari']:.4f}  Cluster Acc: {c['cluster_accuracy']:.4f}")
        print(f"  Cluster mapping: {c['mapping']}")
    if "engram_cos" in results["model3"]:
        ec = results["model3"]["engram_cos"]
        print(f"  Engram cosine sim — within: {ec['within']}  between: {ec['between']}")

    # ── Cross-dataset comparison (load SST-2 results if available) ───────────
    sst2_results_path = "/Data/Code/sentiment/results_7b/results.json"
    if os.path.exists(sst2_results_path):
        print(f"\n{'='*60}")
        print(f"  CROSS-DATASET COMPARISON: SST-2 vs AG News")
        print(f"{'='*60}")
        with open(sst2_results_path) as f:
            sst2 = json.load(f)

        print(f"  {'Metric':<35} {'SST-2':>10} {'AG News':>10}")
        print(f"  {'-'*55}")

        for mt in [1, 2, 3]:
            sst2_acc = sst2[f"model{mt}"]["val"][-1]["accuracy"]
            ag_acc = results[f"model{mt}"]["val"][-1]["accuracy"]
            print(f"  Model {mt} val accuracy            {sst2_acc:>10.4f} {ag_acc:>10.4f}")

        sst2_m1 = sst2["model1"]["val"][-1]["accuracy"]
        sst2_m2 = sst2["model2"]["val"][-1]["accuracy"]
        sst2_m3 = sst2["model3"]["val"][-1]["accuracy"]
        print(f"  Dual-obj gain (M2-M1)            {sst2_m2-sst2_m1:>+10.4f} {m2-m1:>+10.4f}")
        print(f"  Engram delta (M3-M2)             {sst2_m3-sst2_m2:>+10.4f} {m3-m2:>+10.4f}")

        if "cluster" in sst2.get("model3", {}):
            sc = sst2["model3"]["cluster"]
            ac = results["model3"]["cluster"]
            print(f"  Engram ARI                       {sc['ari']:>10.4f} {ac['ari']:>10.4f}")
            print(f"  Engram cluster accuracy           {sc['cluster_accuracy']:>10.4f} {ac['cluster_accuracy']:>10.4f}")
        if "engram_cos" in sst2.get("model3", {}):
            se = sst2["model3"]["engram_cos"]
            ae = results["model3"]["engram_cos"]
            print(f"  Engram cos within-class          {se['within']:>10.4f} {ae['within']:>10.4f}")
            print(f"  Engram cos between-class         {se['between']:>10.4f} {ae['between']:>10.4f}")

    # ── Learning curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for mt in [1, 2, 3]:
        r = results[f"model{mt}"]
        epochs = [v["epoch"] for v in r["val"]]
        axes[0].plot(epochs, [v["accuracy"] for v in r["val"]], "o-",
                     label=f"Model {mt}")
        axes[1].plot(epochs, [v["cls_loss"] for v in r["val"]], "o-",
                     label=f"Model {mt}")
    axes[0].set_title("Val Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Val Classification Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for mt in [2, 3]:
        r = results[f"model{mt}"]
        epochs = [v["epoch"] for v in r["val"]]
        lm = [v["lm_loss"] for v in r["val"]]
        axes[2].plot(epochs, lm, "o-", label=f"Model {mt}")
    axes[2].set_title("Val LM Loss"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle("AG News — Learning Curves", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150,
                bbox_inches="tight")
    print(f"\n  Saved learning_curves.png")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results.json")


if __name__ == "__main__":
    main()

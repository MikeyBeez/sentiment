"""
7B-scale dual-head sentiment experiment with engram feedback.

Three models trained on SST-2 with QLoRA on Qwen2.5-7B:
  1. Sentiment-only (LoRA backbone + sentiment head, no LM loss)
  2. Dual-head (LoRA backbone + sentiment head + LM loss)
  3. Dual-head + engram (same as 2, with 128-dim bottleneck fed back as prefix)

For Model 3, each batch is split in half: first half processes with zero engram,
producing engrams that feed (without detach) into the second half. This gives
the compressor end-to-end gradients through the dual-head loss.

After training, engrams from Model 3 are clustered to check if sentiment
separates naturally in the bottleneck space.
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
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME    = "Qwen/Qwen2.5-7B"
MAX_LENGTH    = 128
BATCH_SIZE    = 8          # per-device batch; Model 3 uses 4 (paired halves)
GRAD_ACCUM    = 2          # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS        = 3
LR            = 2e-4
WARMUP_RATIO  = 0.05
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
ENGRAM_DIM    = 128
SEED          = 42
DEVICE        = "cuda"
OUT_DIR       = "/Data/Code/sentiment/results_7b"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(tokenizer):
    raw = load_dataset("glue", "sst2")

    def tok_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train = raw["train"].map(tok_fn, batched=True)
    val   = raw["validation"].map(tok_fn, batched=True)
    train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train, val

# ── Model wrapper ────────────────────────────────────────────────────────────

class DualHeadWrapper(nn.Module):
    """Wraps a QLoRA CausalLM with sentiment head and optional engram."""

    def __init__(self, peft_model, hidden_size, model_type, pad_token_id):
        super().__init__()
        self.peft_model = peft_model
        self.causal_lm  = peft_model.get_base_model()
        self.model_type  = model_type
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # Sentiment head — full precision
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        ).float()

        # Engram components — full precision
        self.use_engram = (model_type == 3)
        if self.use_engram:
            self.engram_compressor = nn.Sequential(
                nn.Linear(hidden_size, ENGRAM_DIM),
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

        # ── Mean pool for sentiment / engram ─────────────────────────────────
        token_hidden = hidden[:, 1:, :] if has_prefix else hidden
        pool_mask = orig_mask.unsqueeze(-1).float()
        pooled = (token_hidden.float() * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)

        # ── Sentiment head (fp32) ────────────────────────────────────────────
        sent_logits = self.sentiment_head(pooled)
        sent_loss = F.cross_entropy(sent_logits, labels) if labels is not None else None

        # ── Engram compressor (fp32) ─────────────────────────────────────────
        new_engram = None
        if self.use_engram:
            new_engram = self.engram_compressor(pooled)

        # ── Combined loss ────────────────────────────────────────────────────
        if self.model_type == 1:
            total_loss = sent_loss
        else:
            total_loss = 0.5 * lm_loss + 0.5 * sent_loss

        return {
            "loss": total_loss,
            "sentiment_logits": sent_logits,
            "sentiment_loss": sent_loss,
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
    wrapper.sentiment_head.to(DEVICE)
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
    heads = {"sentiment_head": wrapper.sentiment_head.state_dict()}
    if model_type == 3:
        heads["engram_compressor"] = wrapper.engram_compressor.state_dict()
        heads["engram_projector"]  = wrapper.engram_projector.state_dict()
    torch.save(heads, os.path.join(ckpt_dir, "heads.pt"))
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model_type, train_ds, val_ds, tokenizer):
    model_names = {1: "Sentiment Only", 2: "Dual Head", 3: "Dual + Engram"}
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
        epoch_loss, epoch_sent, epoch_lm = 0.0, 0.0, 0.0
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
                    sl = out1["sentiment_loss"].item() + out2["sentiment_loss"].item()
                    epoch_sent += sl / 2
                    if out1["lm_loss"] is not None:
                        epoch_lm += (out1["lm_loss"].item() + out2["lm_loss"].item()) / 2
                    p1 = out1["sentiment_logits"].argmax(-1)
                    p2 = out2["sentiment_logits"].argmax(-1)
                    epoch_correct += (p1 == labels[:half]).sum().item()
                    epoch_correct += (p2 == labels[half:2*half]).sum().item()
                    epoch_total += 2 * half
                    epoch_loss += (out1["loss"].item() + out2["loss"].item()) / 2
            else:
                out = wrapper(ids, mask, labels)
                loss = out["loss"] / ga
                loss.backward()

                with torch.no_grad():
                    epoch_sent += out["sentiment_loss"].item()
                    if out["lm_loss"] is not None:
                        epoch_lm += out["lm_loss"].item()
                    preds = out["sentiment_logits"].argmax(-1)
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
            "sent_loss": round(epoch_sent / n, 4),
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
              f"val_sent_loss={val_metrics['sent_loss']:.4f}"
              + (f"  val_lm_loss={val_metrics['lm_loss']:.4f}" if val_metrics['lm_loss'] else ""))

        save_checkpoint(wrapper, epoch, model_type, all_metrics)

    return wrapper, all_metrics, val_dl

# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(wrapper, val_dl, model_type, collect_engrams=False):
    wrapper.eval()
    correct, total = 0, 0
    total_sent, total_lm = 0.0, 0.0
    all_engrams, all_labels = [], []
    # For engram cosine similarity tracking
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

        preds = out["sentiment_logits"].argmax(-1)
        correct += (preds == labels).sum().item()
        total += B
        total_sent += out["sentiment_loss"].item()
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
        "sent_loss": round(total_sent / n, 4),
        "lm_loss": round(total_lm / n, 4) if model_type != 1 else None,
    }

    if collect_engrams and all_engrams:
        engrams_np = np.concatenate(all_engrams)
        labels_np  = np.concatenate(all_labels)
        eng_t = torch.cat(all_engram_tensors)
        lab_t = torch.cat(all_label_tensors)

        # Cosine similarity: within-class vs cross-class
        normed = F.normalize(eng_t, dim=1)
        cos = normed @ normed.T
        same = lab_t.unsqueeze(0) == lab_t.unsqueeze(1)
        eye  = torch.eye(len(lab_t), dtype=torch.bool)
        same_no_diag = same & ~eye
        diff = ~same

        metrics["engram_cos_within"]  = round(cos[same_no_diag].mean().item(), 4)
        metrics["engram_cos_between"] = round(cos[diff].mean().item(), 4)
        return metrics, engrams_np, labels_np

    return metrics

# ── Clustering analysis ──────────────────────────────────────────────────────

def cluster_analysis(engrams, labels):
    print(f"\n{'='*60}")
    print(f"  ENGRAM CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    print(f"  Engram shape: {engrams.shape}")

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(engrams)-1))
    proj = tsne.fit_transform(engrams)

    km = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    clusters = km.fit_predict(engrams)

    ari = adjusted_rand_score(labels, clusters)
    acc1 = accuracy_score(labels, clusters)
    acc2 = accuracy_score(labels, 1 - clusters)
    cluster_acc = max(acc1, acc2)

    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Cluster accuracy:    {cluster_acc:.4f}")

    if ari > 0.1:
        print("  -> Engrams naturally separate positive/negative!")
    elif ari > 0.02:
        print("  -> Weak sentiment structure in engram space.")
    else:
        print("  -> No clear sentiment separation.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#e74c3c" if l == 0 else "#2ecc71" for l in labels]
    ax1.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.5, s=10)
    ax1.set_title("Engram t-SNE — TRUE sentiment\n(red=neg, green=pos)")
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")

    cc = ["#3498db" if c == 0 else "#e67e22" for c in clusters]
    ax2.scatter(proj[:, 0], proj[:, 1], c=cc, alpha=0.5, s=10)
    ax2.set_title(f"Engram t-SNE — K-Means\nARI={ari:.3f}, Acc={cluster_acc:.3f}")
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "engram_clusters.png"), dpi=150)
    print(f"  Saved engram_clusters.png")

    return {"ari": round(ari, 4), "cluster_accuracy": round(cluster_acc, 4)}

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
    print("Loading SST-2...")
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
    print(f"  FINAL RESULTS")
    print(f"{'='*60}")

    for mt in [1, 2, 3]:
        key = f"model{mt}"
        val = results[key]["val"][-1]
        print(f"  Model {mt}: val_acc={val['accuracy']:.4f}  "
              f"sent_loss={val['sent_loss']:.4f}"
              + (f"  lm_loss={val['lm_loss']:.4f}" if val.get("lm_loss") else ""))

    m1 = results["model1"]["val"][-1]["accuracy"]
    m2 = results["model2"]["val"][-1]["accuracy"]
    m3 = results["model3"]["val"][-1]["accuracy"]

    print(f"\n  Model 2 vs 1 (dual-objective regularization): {m2-m1:+.4f}")
    print(f"  Model 3 vs 2 (engram feedback):               {m3-m2:+.4f}")

    if "cluster" in results["model3"]:
        c = results["model3"]["cluster"]
        print(f"  Engram ARI: {c['ari']:.4f}  Cluster Acc: {c['cluster_accuracy']:.4f}")
    if "engram_cos" in results["model3"]:
        ec = results["model3"]["engram_cos"]
        print(f"  Engram cosine sim — within: {ec['within']}  between: {ec['between']}")

    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for mt in [1, 2, 3]:
        r = results[f"model{mt}"]
        epochs = [v["epoch"] for v in r["val"]]
        axes[0].plot(epochs, [v["accuracy"] for v in r["val"]], "o-",
                     label=f"Model {mt}")
        axes[1].plot(epochs, [v["sent_loss"] for v in r["val"]], "o-",
                     label=f"Model {mt}")
    axes[0].set_title("Val Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Val Sentiment Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    for mt in [2, 3]:
        r = results[f"model{mt}"]
        epochs = [v["epoch"] for v in r["val"]]
        lm = [v["lm_loss"] for v in r["val"]]
        axes[2].plot(epochs, lm, "o-", label=f"Model {mt}")
    axes[2].set_title("Val LM Loss"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150)
    print(f"\n  Saved learning_curves.png")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved results.json")


if __name__ == "__main__":
    main()

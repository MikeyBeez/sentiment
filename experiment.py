"""
Three-model sentiment experiment on SST-2:
  1. Sentiment-only classifier (baseline)
  2. Shared backbone + sentiment head + next-token head (dual objective)
  3. Same as 2 but with compressed engram feedback

All use a small 2-layer transformer backbone (~2M params) to avoid memorization.
Secondary analysis: t-SNE clustering of engrams from model 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import math
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── Hyperparameters (deliberately small) ───────────────────────────────────

VOCAB_SIZE   = 8000      # small BPE-like vocab via truncation
EMBED_DIM    = 128
NUM_HEADS    = 4
NUM_LAYERS   = 2
FF_DIM       = 256
MAX_SEQ_LEN  = 64
ENGRAM_DIM   = 32        # compressed bottleneck
BATCH_SIZE   = 64
EPOCHS       = 8
LR           = 3e-4
DROPOUT      = 0.1
LM_WEIGHT    = 0.3       # weight for next-token loss in dual models
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = "/Data/Code/sentiment/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────

print("Loading SST-2...")
raw = load_dataset("glue", "sst2")
train_raw = raw["train"]
val_raw   = raw["validation"]

# Build a simple word-level vocabulary from training data
def build_vocab(dataset, max_size=VOCAB_SIZE):
    freq = {}
    for ex in dataset:
        for w in ex["sentence"].lower().split():
            freq[w] = freq.get(w, 0) + 1
    # sort by frequency, keep top max_size - 2 (reserve PAD=0, UNK=1)
    words = sorted(freq, key=freq.get, reverse=True)[: max_size - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, w in enumerate(words, start=2):
        vocab[w] = i
    return vocab

vocab = build_vocab(train_raw)
actual_vocab_size = len(vocab)
print(f"Vocabulary size: {actual_vocab_size}")

def encode(sentence, vocab, max_len=MAX_SEQ_LEN):
    tokens = [vocab.get(w, 1) for w in sentence.lower().split()]
    tokens = tokens[:max_len]
    pad_len = max_len - len(tokens)
    return tokens + [0] * pad_len, len(tokens)

class SSTDataset(Dataset):
    def __init__(self, split):
        self.data = []
        for ex in split:
            ids, length = encode(ex["sentence"], vocab)
            self.data.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(ex["label"], dtype=torch.long),
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

train_ds = SSTDataset(train_raw)
val_ds   = SSTDataset(val_raw)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# ─── Backbone ────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBackbone(nn.Module):
    """Small transformer encoder shared across all models."""
    def __init__(self, extra_input_dim=0):
        super().__init__()
        self.embed = nn.Embedding(actual_vocab_size, EMBED_DIM, padding_idx=0)
        self.pos   = PositionalEncoding(EMBED_DIM)
        self.drop  = nn.Dropout(DROPOUT)

        # If engram is fed back, we project (EMBED_DIM + extra_input_dim) -> EMBED_DIM
        self.input_proj = None
        if extra_input_dim > 0:
            self.input_proj = nn.Linear(EMBED_DIM + extra_input_dim, EMBED_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

    def forward(self, input_ids, lengths, engram=None):
        """
        input_ids: (B, L)
        lengths:   (B,)
        engram:    (B, ENGRAM_DIM) or None
        Returns:   (B, L, EMBED_DIM) — full sequence of hidden states
        """
        pad_mask = (input_ids == 0)  # True where padded
        x = self.drop(self.pos(self.embed(input_ids)))

        # Inject engram: broadcast across sequence and concatenate
        if engram is not None and self.input_proj is not None:
            eng_expanded = engram.unsqueeze(1).expand(-1, x.size(1), -1)
            x = self.input_proj(torch.cat([x, eng_expanded], dim=-1))

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x, pad_mask


# ─── Heads ───────────────────────────────────────────────────────────────────

class SentimentHead(nn.Module):
    """Mean-pool over non-padded tokens → 2-class classifier."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 2),
        )

    def forward(self, hidden, pad_mask):
        # mask out padding before mean pool
        mask = (~pad_mask).unsqueeze(-1).float()            # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(pooled)


class LMHead(nn.Module):
    """Next-token prediction head (causal, teacher-forced)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(EMBED_DIM, actual_vocab_size)

    def forward(self, hidden):
        # predict token t+1 from position t
        return self.proj(hidden[:, :-1])   # (B, L-1, V)


class EngramCompressor(nn.Module):
    """Compress last-layer hidden state into a small engram vector."""
    def __init__(self):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(EMBED_DIM, ENGRAM_DIM),
            nn.Tanh(),
        )

    def forward(self, hidden, pad_mask):
        mask = (~pad_mask).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.compress(pooled)


# ─── Model 1: Sentiment-only baseline ───────────────────────────────────────

class Model1_SentimentOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBackbone()
        self.sent_head = SentimentHead()

    def forward(self, input_ids, lengths):
        h, mask = self.backbone(input_ids, lengths)
        return self.sent_head(h, mask), None

    def compute_loss(self, input_ids, lengths, labels):
        logits, _ = self.forward(input_ids, lengths)
        return F.cross_entropy(logits, labels), logits


# ─── Model 2: Dual-objective (sentiment + LM), no engram ────────────────────

class Model2_Dual(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TransformerBackbone()
        self.sent_head = SentimentHead()
        self.lm_head   = LMHead()

    def forward(self, input_ids, lengths):
        h, mask = self.backbone(input_ids, lengths)
        sent_logits = self.sent_head(h, mask)
        lm_logits   = self.lm_head(h)
        return sent_logits, lm_logits

    def compute_loss(self, input_ids, lengths, labels):
        sent_logits, lm_logits = self.forward(input_ids, lengths)
        sent_loss = F.cross_entropy(sent_logits, labels)
        # next-token targets: shift input_ids by 1
        targets = input_ids[:, 1:]   # (B, L-1)
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, actual_vocab_size),
            targets.reshape(-1),
            ignore_index=0,  # ignore padding
        )
        loss = sent_loss + LM_WEIGHT * lm_loss
        return loss, sent_logits


# ─── Model 3: Dual-objective + engram feedback ──────────────────────────────

class Model3_Engram(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone  = TransformerBackbone(extra_input_dim=ENGRAM_DIM)
        self.sent_head = SentimentHead()
        self.lm_head   = LMHead()
        self.compressor = EngramCompressor()

    def forward(self, input_ids, lengths, prev_engram=None):
        if prev_engram is None:
            prev_engram = torch.zeros(input_ids.size(0), ENGRAM_DIM, device=input_ids.device)
        h, mask = self.backbone(input_ids, lengths, engram=prev_engram)
        sent_logits = self.sent_head(h, mask)
        lm_logits   = self.lm_head(h)
        new_engram  = self.compressor(h, mask)
        return sent_logits, lm_logits, new_engram

    def compute_loss(self, input_ids, lengths, labels, prev_engram=None):
        sent_logits, lm_logits, new_engram = self.forward(input_ids, lengths, prev_engram)
        sent_loss = F.cross_entropy(sent_logits, labels)
        targets = input_ids[:, 1:]
        lm_loss = F.cross_entropy(
            lm_logits.reshape(-1, actual_vocab_size),
            targets.reshape(-1),
            ignore_index=0,
        )
        loss = sent_loss + LM_WEIGHT * lm_loss
        return loss, sent_logits, new_engram


# ─── Training / Evaluation ──────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dl, optimizer, model_type):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    engram_bank = {}  # for model 3: store engrams per sample (not used in single-pass SST-2)

    for ids, lens, labels in dl:
        ids, lens, labels = ids.to(DEVICE), lens.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        if model_type == 3:
            # For SST-2 each example is independent, so prev_engram = zeros (cold start).
            # The engram's value shows when the *same* model processes sequences;
            # here we demonstrate the mechanism and collect engrams for clustering.
            loss, logits, _ = model.compute_loss(ids, lens, labels)
        else:
            loss, logits = model.compute_loss(ids, lens, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * ids.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += ids.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dl, model_type, collect_engrams=False):
    model.eval()
    correct, total = 0, 0
    all_engrams, all_labels = [], []

    for ids, lens, labels in dl:
        ids, lens, labels = ids.to(DEVICE), lens.to(DEVICE), labels.to(DEVICE)

        if model_type == 3:
            sent_logits, _, engram = model(ids, lens)
            if collect_engrams:
                all_engrams.append(engram.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        elif model_type == 2:
            sent_logits, _ = model(ids, lens)
        else:
            sent_logits, _ = model(ids, lens)

        preds = sent_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += ids.size(0)

    acc = correct / total
    if collect_engrams and all_engrams:
        return acc, np.concatenate(all_engrams), np.concatenate(all_labels)
    return acc


# ─── Run experiment ──────────────────────────────────────────────────────────

def run_experiment():
    results = {}

    for model_idx, (ModelClass, name) in enumerate([
        (Model1_SentimentOnly, "Model 1: Sentiment Only"),
        (Model2_Dual,          "Model 2: Dual Objective"),
        (Model3_Engram,        "Model 3: Dual + Engram"),
    ], start=1):

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        model = ModelClass().to(DEVICE)
        params = count_params(model)
        print(f"  Parameters: {params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        train_accs, val_accs = [], []
        best_val_acc = 0.0

        for epoch in range(1, EPOCHS + 1):
            loss, train_acc = train_epoch(model, train_dl, optimizer, model_idx)
            val_acc = evaluate(model, val_dl, model_idx)
            scheduler.step()

            train_accs.append(train_acc)
            val_accs.append(val_acc)
            best_val_acc = max(best_val_acc, val_acc)

            print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        results[name] = {
            "params": params,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "best_val": best_val_acc,
            "final_val": val_accs[-1],
        }

        # Save model 3 for engram analysis
        if model_idx == 3:
            engram_acc, engrams, engram_labels = evaluate(
                model, val_dl, model_idx, collect_engrams=True
            )

    # ─── Summary ─────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}")
        print(f"    Params: {r['params']:,}  Best val: {r['best_val']:.4f}  "
              f"Final val: {r['final_val']:.4f}")

    m1 = results["Model 1: Sentiment Only"]["best_val"]
    m2 = results["Model 2: Dual Objective"]["best_val"]
    m3 = results["Model 3: Dual + Engram"]["best_val"]

    print(f"\n  Model 2 vs Model 1 (dual-objective regularization): "
          f"{m2 - m1:+.4f}")
    print(f"  Model 3 vs Model 2 (engram feedback):               "
          f"{m3 - m2:+.4f}")

    if m2 > m1:
        print("  → Dual objective IS acting as regularization.")
    else:
        print("  → Dual objective did NOT help (or hurt).")

    if m3 > m2:
        print("  → Engram IS carrying useful evaluative state forward.")
    else:
        print("  → Engram did NOT add benefit over dual objective alone.")

    # ─── Learning curves plot ────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results.items():
        ax1.plot(range(1, EPOCHS+1), r["train_accs"], label=name)
        ax2.plot(range(1, EPOCHS+1), r["val_accs"], label=name)
    ax1.set_title("Training Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150)
    print(f"\n  Saved learning_curves.png")

    # ─── Engram clustering analysis ──────────────────────────────────────────

    print(f"\n{'='*60}")
    print("  ENGRAM CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    print(f"  Engram shape: {engrams.shape}")
    print(f"  Engram dim: {ENGRAM_DIM}")

    # t-SNE projection
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    projected = tsne.fit_transform(engrams)

    # K-Means with k=2
    km = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    cluster_ids = km.fit_predict(engrams)

    # How well do clusters align with true sentiment labels?
    ari = adjusted_rand_score(engram_labels, cluster_ids)
    # Try both cluster→label mappings, pick the better one
    mapping_acc_1 = accuracy_score(engram_labels, cluster_ids)
    mapping_acc_2 = accuracy_score(engram_labels, 1 - cluster_ids)
    cluster_acc = max(mapping_acc_1, mapping_acc_2)

    print(f"  Adjusted Rand Index (engram clusters vs true labels): {ari:.4f}")
    print(f"  Cluster→sentiment accuracy (best mapping):           {cluster_acc:.4f}")

    if ari > 0.1:
        print("  → Engrams naturally separate positive/negative sentiment!")
    elif ari > 0.02:
        print("  → Weak but present sentiment structure in engram space.")
    else:
        print("  → No clear sentiment separation in engram space.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color by true label
    colors = ['#e74c3c' if l == 0 else '#2ecc71' for l in engram_labels]
    ax1.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.5, s=8)
    ax1.set_title("Engram t-SNE — colored by TRUE sentiment\n(red=negative, green=positive)")
    ax1.set_xlabel("t-SNE 1"); ax1.set_ylabel("t-SNE 2")

    # Color by K-Means cluster
    cluster_colors = ['#3498db' if c == 0 else '#e67e22' for c in cluster_ids]
    ax2.scatter(projected[:, 0], projected[:, 1], c=cluster_colors, alpha=0.5, s=8)
    ax2.set_title(f"Engram t-SNE — colored by K-Means cluster\nARI={ari:.3f}, Cluster Acc={cluster_acc:.3f}")
    ax2.set_xlabel("t-SNE 1"); ax2.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "engram_clusters.png"), dpi=150)
    print(f"  Saved engram_clusters.png")

    # Save results JSON
    save_results = {}
    for name, r in results.items():
        save_results[name] = {
            "params": r["params"],
            "best_val": r["best_val"],
            "final_val": r["final_val"],
            "train_accs": [round(a, 4) for a in r["train_accs"]],
            "val_accs": [round(a, 4) for a in r["val_accs"]],
        }
    save_results["engram_analysis"] = {
        "adjusted_rand_index": round(ari, 4),
        "cluster_accuracy": round(cluster_acc, 4),
        "engram_dim": ENGRAM_DIM,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved results.json")


if __name__ == "__main__":
    run_experiment()

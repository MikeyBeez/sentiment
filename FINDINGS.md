# Dual-Objective Training and Engram Feedback for Sentiment Classification

## Overview

This study investigates whether auxiliary training objectives and compressed hidden-state feedback (engrams) improve sentiment classification in transformer models. We compare three architectures at two scales on the Stanford Sentiment Treebank (SST-2) binary classification task.

### Models

| # | Architecture | Description |
|---|---|---|
| 1 | **Sentiment Only** | Baseline — backbone trained with a single sentiment classification head |
| 2 | **Dual Head** | Shared backbone with both a sentiment head and a next-token prediction (LM) head; combined loss = 0.5 * LM + 0.5 * sentiment |
| 3 | **Dual + Engram** | Same as Model 2, plus a compressed bottleneck representation ("engram") from the last hidden layer, fed back as a learned prefix embedding on the next input |

### Hypotheses

- **H1**: If Model 2 > Model 1, the dual objective acts as a regularizer, keeping backbone representations general rather than collapsing onto the classification task.
- **H2**: If Model 3 > Model 2, the engram carries useful evaluative state forward, providing the model with a compressed memory of prior context.
- **H3**: Engram representations will naturally separate by sentiment polarity in the bottleneck space, even without an explicit clustering objective.

---

## Experiment 1: Small Model (from scratch)

### Setup

- **Backbone**: 2-layer transformer, 128-dim embeddings, 4 attention heads (~1.3M parameters)
- **Vocabulary**: Word-level, built from training data
- **Engram dimension**: 32
- **Training**: 8 epochs, Adam optimizer, lr=3e-4, batch size=64
- **Dataset**: SST-2 (67,349 train / 872 validation)

### Results

| Model | Best Val Accuracy | Final Val Accuracy |
|---|:-:|:-:|
| Model 1: Sentiment Only | 80.28% | 79.82% |
| Model 2: Dual Head | **80.96%** | **80.96%** |
| Model 3: Dual + Engram | 79.36% | 79.36% |

**Model 2 vs 1: +0.69%** — The dual objective provides a small regularization benefit.

**Model 3 vs 2: -1.61%** — The engram feedback hurt performance. At this scale the backbone has limited capacity, and the engram prefix likely introduces noise that competes with the model's ability to attend to the actual input tokens.

### Engram Clustering

| Metric | Value |
|---|:-:|
| Adjusted Rand Index | 0.236 |
| Cluster Accuracy | 74.3% |

The engram representations show weak but above-chance separation of positive and negative sentiment. The t-SNE visualization reveals overlapping clusters — the 32-dimensional bottleneck captures some sentiment signal but is far from cleanly organized.

### Learning Curves

![Small model learning curves](results/learning_curves.png)

Training accuracy for Model 1 rises fastest (fewer parameters to coordinate), while Models 2 and 3 train more slowly due to the shared LM objective. Validation curves converge to similar ranges, with Model 2 slightly ahead.

### Engram t-SNE

![Small model engram clusters](results/engram_clusters.png)

Left: colored by true sentiment label. Right: colored by K-Means cluster assignment. The clusters overlap substantially — the small model's engram captures sentiment only weakly.

---

## Experiment 2: 7B Pretrained Model (QLoRA fine-tuning)

### Setup

- **Base model**: Qwen2.5-7B
- **Quantization**: 4-bit NF4 with double quantization (BitsAndBytes)
- **Fine-tuning**: LoRA rank=16, alpha=32, applied to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Sentiment head**: Linear(3584→256) → GELU → Dropout(0.1) → Linear(256→2), trained in fp32
- **Engram compressor**: Linear(3584→128) → Tanh, trained in fp32
- **Engram projector**: Linear(128→3584), maps engram back to embedding space as a prefix token
- **Training**: 3 epochs, AdamW, lr=2e-4, cosine schedule with warmup, gradient checkpointing, batch size=8 (Model 3 uses 4 with sequential pair processing)
- **Combined loss**: 0.5 * LM_loss + 0.5 * sentiment_loss (Models 2 and 3)
- **VRAM**: Fits within 16GB (RTX 4080)
- **Dataset**: SST-2 (67,349 train / 872 validation)
- **Trainable parameters**: ~41M / 4.4B total (0.94%)

### Engram Training Strategy

For Model 3, each batch is split into sequential pairs. The first half of the batch is processed with a zero-initialized engram, producing a compressed engram output. This engram (not detached from the computation graph) is then fed as a prefix embedding to the second half of the batch. This allows gradients from the second half's loss to flow back through the engram compressor, training it end-to-end.

### Results

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Best | Time |
|---|:-:|:-:|:-:|:-:|:-:|
| Model 1: Sentiment Only | 94.38% | 95.41% | **96.22%** | **96.22%** | 380 min |
| Model 2: Dual Head | 95.87% | **96.79%** | 96.56% | **96.79%** | 417 min |
| Model 3: Dual + Engram | 95.53% | **96.67%** | 96.67% | **96.67%** | 635 min |

**Model 2 vs 1: +0.57%** — The dual objective again provides a regularization benefit. Model 2 peaks at epoch 2 and begins to overfit slightly at epoch 3, while Model 1 is still improving — suggesting the LM head helps the model converge faster to a better representation.

**Model 3 vs 2: -0.12%** — The engram provides negligible benefit to accuracy. The pretrained 7B backbone already carries rich contextual representations; the engram prefix adds little information that isn't already available.

### Training Dynamics

| Model | Train Loss (E3) | Train Sent Loss (E3) | Train LM Loss (E3) | Train Acc (E3) |
|---|:-:|:-:|:-:|:-:|
| Model 1 | 0.058 | 0.058 | — | 98.45% |
| Model 2 | 0.458 | 0.033 | 0.883 | 98.80% |
| Model 3 | 0.793 | 0.034 | 1.551 | 98.75% |

All models achieve >98% training accuracy by epoch 3. Model 3 has higher total loss due to the additional challenge of predicting next tokens with the engram prefix, but its sentiment-specific loss (0.034) is comparable to Model 2 (0.033).

Validation LM loss increases across epochs for both Models 2 and 3 (3.88 → 4.69), indicating the LM head overfits to training data patterns even as sentiment accuracy improves. This is expected — SST-2 sentences are short and stylistically narrow, so the LM head memorizes domain-specific patterns rather than learning generalizable language modeling.

### Learning Curves

![7B model learning curves](results_7b/learning_curves.png)

Left panel: All three models converge to similar validation accuracy (96-97%), with Model 2 reaching peak performance fastest. Center panel: Sentiment loss drops steadily for all models, with Model 3 achieving the lowest validation sentiment loss. Right panel: LM loss increases on validation data across epochs, suggesting the LM head overfits.

### Engram Clustering

| Metric | Value |
|---|:-:|
| Adjusted Rand Index | 0.833 |
| Cluster Accuracy | 95.64% |
| Within-class cosine similarity | 0.953 |
| Between-class cosine similarity | 0.838 |

![7B model engram clusters](results_7b/engram_clusters.png)

Left: colored by true sentiment. Right: colored by K-Means cluster. The 128-dimensional engram representations form two well-separated clusters that align almost perfectly with sentiment polarity. The engram bottleneck has learned to compress the 3,584-dimensional hidden state into a representation where sentiment is the dominant organizing principle — without any explicit clustering loss.

The high within-class cosine similarity (0.953) indicates that engrams for same-sentiment inputs are tightly grouped, while the between-class similarity (0.838) shows meaningful but not maximal separation — consistent with sentiment being the primary but not sole axis of variation in the compressed space.

---

## Cross-Scale Comparison

### Accuracy

| Model | Small (~1.3M) | 7B (QLoRA) | Improvement |
|---|:-:|:-:|:-:|
| Model 1 | 80.28% | 96.22% | +15.94% |
| Model 2 | 80.96% | 96.79% | +15.83% |
| Model 3 | 79.36% | 96.67% | +17.31% |

The pretrained 7B model provides a ~16 percentage point improvement across all architectures. The gap is slightly larger for Model 3 (+17.31%), suggesting that the engram mechanism benefits more from rich pretrained representations than from raw capacity.

### Dual-Objective Effect

| Scale | Delta (Model 2 - Model 1) |
|---|:-:|
| Small | +0.69% |
| 7B | +0.57% |

The regularization benefit of the dual objective is consistent across scales — approximately +0.6% in both cases. This suggests the mechanism (preventing representational collapse by maintaining general-purpose features) operates independently of model capacity.

### Engram Effect

| Scale | Delta (Model 3 - Model 2) |
|---|:-:|
| Small | -1.61% |
| 7B | -0.12% |

The engram hurts at both scales, but the penalty diminishes dramatically with model size. The small model loses 1.6% — a significant degradation suggesting the engram prefix disrupts the attention patterns of a capacity-limited backbone. The 7B model barely notices the engram (-0.12%), likely because its pretrained attention mechanisms are robust enough to integrate or ignore the prefix as needed.

### Engram Representation Quality

| Metric | Small (32-dim) | 7B (128-dim) |
|---|:-:|:-:|
| Adjusted Rand Index | 0.236 | **0.833** |
| Cluster Accuracy | 74.3% | **95.6%** |

This is the most dramatic difference between scales. The 7B model's engrams are 3.5x better organized (by ARI) and achieve near-perfect cluster accuracy. The pretrained backbone provides the engram compressor with rich, semantically structured hidden states. Compressing from 3,584 to 128 dimensions is a relatively mild bottleneck for representations that are already well-organized; compressing from 128 to 32 dimensions in the small model, where representations are learned from scratch and still forming, discards far more information.

---

## Conclusions

1. **Dual-objective training provides consistent regularization.** Adding a next-token prediction head alongside the sentiment classifier improves accuracy at both scales (+0.6-0.7%). The LM objective prevents the backbone from overfitting to the classification task, maintaining richer internal representations.

2. **Engram feedback does not improve classification accuracy.** At neither scale does the compressed hidden-state feedback help the model make better sentiment predictions. The backbone's own hidden states already carry the relevant information; the engram prefix is redundant for the classification objective.

3. **Engrams spontaneously encode sentiment.** Despite providing no accuracy benefit, the engram bottleneck learns to organize its compressed representations around sentiment polarity. This effect is weak in the small model (ARI=0.236) but dramatic in the 7B model (ARI=0.833). The bottleneck acts as an information filter that naturally preserves the most task-relevant dimension of variation.

4. **Pretrained representations dramatically improve engram quality.** The jump from 74.3% to 95.6% cluster accuracy suggests that engrams are most useful not as feedback mechanisms during inference, but as interpretable compressed summaries of model state. A well-trained engram bottleneck could serve as a lightweight probe for what the model has "understood" about an input.

5. **The engram's value may lie in interpretability, not performance.** While the engram doesn't help the model classify better, its compressed representation provides a low-dimensional, human-inspectable summary of the model's internal state that cleanly separates sentiment. This could be valuable for model monitoring, debugging, or as a compact feature for downstream systems.

---

## Reproduction

```bash
# Small model experiment (~30 minutes on GPU)
python experiment.py

# 7B QLoRA experiment (~10.5 hours on RTX 4080 16GB)
python experiment_7b.py
```

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- transformers, peft, bitsandbytes, datasets
- scikit-learn, matplotlib, numpy

### Hardware

- Small model: Any GPU with 2GB+ VRAM (or CPU)
- 7B model: GPU with 16GB+ VRAM (tested on RTX 4080)

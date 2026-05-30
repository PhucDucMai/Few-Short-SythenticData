"""
train_clip_fewshot.py

Implementation following:
"Provably Improving Generalization of Few-Shot Models with Synthetic Data"
(ICML 2025)

Algorithm (Lightweight version - Algorithm 2 in paper):
  Phase 1 - Partition Optimization: K-means clustering on CLIP features (fixed, pre-training)
  Phase 2 - Model Optimization: Train with combined loss:
      L = λ·F(S,h) + F(G,h)
        + λ1 * Σ_i (g_i/g) * (1/|G_i||S_i|) * Σ_{s∈S_i,g∈G_i} ||h(s)-h(g)||  [discrepancy]
        + λ2 * (1/g) * Σ_i Σ_{g1,g2∈G_i} (1/g_i) * ||h(g1)-h(g2)||              [robustness]

Key differences from naive implementation:
  - CutMix + Mixup augmentation on all data
  - Fixed clustering done ONCE before training (on normalized CLIP features)
  - Each sample has a pre-assigned cluster ID used throughout training
  - Discrepancy: per-cluster mean distance between real and fake embeddings
  - Robustness: per-cluster mean pairwise distance among fake embeddings
  - All metrics + plots saved after training

Dataset structure:
  train/Real-Img/class_name/*.jpg   (few-shot real images)
  train/Fake-Img/class_name/*.jpg   (synthetic images)
  val/class_name/*.jpg

Note: class folder names may use spaces (val/test) or underscores (train).
      The code normalizes both to a canonical key for matching.
"""

import os
import random
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip
from peft import LoraConfig, get_peft_model
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val_acc has not improved by `min_delta`
    for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = 0.0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        """Call after each epoch. Returns True if training should stop."""
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def normalize_class_name(name: str) -> str:
    """Normalize class name: lowercase + replace spaces/underscores uniformly."""
    return name.lower().replace("_", " ").strip()


class FolderDataset(Dataset):
    """
    Loads images from  root/class_name/*.jpg
    Returns dict with keys: image, label, idx, global_idx (set later).
    """

    def __init__(self, root: str, class_to_idx: dict, transform, source: str):
        self.samples = []
        self.transform = transform
        self.source = source

        norm_to_idx = {normalize_class_name(k): v for k, v in class_to_idx.items()}

        for cls_dir_name in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls_dir_name)
            if not os.path.isdir(cls_path):
                continue

            norm_name = normalize_class_name(cls_dir_name)
            if norm_name not in norm_to_idx:
                print(f"[WARNING] '{cls_dir_name}' not found in class_to_idx, skipping.")
                continue

            label = norm_to_idx[norm_name]

            for f in sorted(os.listdir(cls_path)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append({
                        "path": os.path.join(cls_path, f),
                        "label": label,
                        "source": source,
                    })

        print(f"[Dataset:{source}] {len(self.samples)} images from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        img = Image.open(item["path"]).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "label": item["label"],
            "idx": idx,       # local index within this dataset
            "source": item["source"],
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, feat_dim: int, num_classes: int):
        super().__init__()
        self.clip_model = clip_model
        self.head = nn.Linear(feat_dim, num_classes)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.clip_model.encode_image(images)
        return F.normalize(feat.float(), dim=-1)

    def forward(self, images: torch.Tensor):
        feat = self.encode(images)
        logits = self.head(feat)
        return feat, logits


# ---------------------------------------------------------------------------
# Feature extraction (no grad, for clustering)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: CLIPClassifier, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    feats = []
    for batch in tqdm(loader, desc="  extract_features", leave=False):
        imgs = batch["image"].to(device)
        feat = model.encode(imgs)
        feats.append(feat.cpu().float())
    return torch.cat(feats, dim=0).numpy()


# ---------------------------------------------------------------------------
# Augmentation helpers: CutMix + Mixup  (paper Appendix B)
# ---------------------------------------------------------------------------

def rand_bbox(size, lam):
    """CutMix bounding box."""
    W, H = size[-1], size[-2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    rand_idx = torch.randperm(batch_size, device=images.device)
    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images_mix = images.clone()
    images_mix[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2))
    labels_a = labels
    labels_b = labels[rand_idx]
    return images_mix, labels_a, labels_b, lam_actual


def apply_mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    rand_idx = torch.randperm(batch_size, device=images.device)
    images_mix = lam * images + (1 - lam) * images[rand_idx]
    labels_a = labels
    labels_b = labels[rand_idx]
    return images_mix, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Loss components  (Equation 7 in paper)
# ---------------------------------------------------------------------------

def compute_discrepancy_loss(
    real_feat_by_cluster: dict,   # cid -> (N_r, D) tensor
    fake_feat_by_cluster: dict,   # cid -> (N_f, D) tensor
    total_fake: int,
    device: str,
) -> torch.Tensor:
    """
    A1 = Σ_i (g_i/g) * d_h(G_i, S_i)
       = Σ_i (g_i/g) * mean_{s∈S_i, g∈G_i} ||h(s) - h(g)||_2
    """
    loss = torch.tensor(0.0, device=device)
    for cid in real_feat_by_cluster:
        if cid not in fake_feat_by_cluster:
            continue
        r_feat = real_feat_by_cluster[cid]   # (N_r, D)
        f_feat = fake_feat_by_cluster[cid]   # (N_f, D)
        if r_feat.size(0) == 0 or f_feat.size(0) == 0:
            continue
        # mean over all (real, fake) pairs in cluster  →  ||mean_r - mean_f||
        # equivalent to averaging pairwise distances via mean centers
        r_center = r_feat.mean(0)
        f_center = f_feat.mean(0)
        weight = f_feat.size(0) / max(total_fake, 1)
        loss = loss + weight * torch.norm(r_center - f_center, p=2)
    return loss


def compute_robustness_loss(
    fake_feat_by_cluster: dict,   # cid -> (N_f, D) tensor
    real_count_by_cluster: dict,  # cid -> int
    total_real: int,
    device: str,
) -> torch.Tensor:
    """
    A2 = (1/g) Σ_i Σ_{g1,g2∈G_i} (1/g_i) ||h(g1) - h(g2)||_2
       weighted by n_i/n (real count ratio per cluster)
    """
    loss = torch.tensor(0.0, device=device)
    for cid, f_feat in fake_feat_by_cluster.items():
        if f_feat.size(0) < 2:
            continue
        dist = torch.cdist(f_feat, f_feat, p=2)   # (N_f, N_f)
        rob = dist.mean()
        n_real_in_cluster = real_count_by_cluster.get(cid, 0)
        weight = n_real_in_cluster / max(total_real, 1)
        loss = loss + weight * rob
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: CLIPClassifier, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        _, logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_metrics(history: dict, output_dir: str):
    """Save training curve plots."""
    epochs = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0][0].plot(epochs, history["loss"], "b-o", markersize=3)
    axes[0][0].set_title("Total Loss")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].grid(True)

    # Regularization terms
    axes[0][1].plot(epochs, history["discrepancy"], "r-o", markersize=3, label="Discrepancy")
    axes[0][1].plot(epochs, history["robustness"],  "g-o", markersize=3, label="Robustness")
    axes[0][1].set_title("Regularization Terms")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Value")
    axes[0][1].legend()
    axes[0][1].grid(True)

    # Validation accuracy
    axes[1][0].plot(epochs, history["val_acc"], "m-o", markersize=3)
    axes[1][0].set_title("Validation Accuracy")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Accuracy (%)")
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc   = max(history["val_acc"])
    axes[1][0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                       label=f"Best: {best_acc:.2f}% @ ep{best_epoch}")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(True)

    # LR schedule
    if "lr" in history and history["lr"]:
        axes[1][1].plot(epochs, history["lr"], "c-o", markersize=3)
        axes[1][1].set_title("Learning Rate Schedule")
        axes[1][1].set_xlabel("Epoch")
        axes[1][1].set_ylabel("LR")
        axes[1][1].set_yscale("log")
        axes[1][1].grid(True)
    else:
        axes[1][1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[Saved] training_curves.png")


def plot_loss_components(history: dict, output_dir: str):
    """Save individual loss component curves."""
    epochs = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    components = [
        ("ce_real", "CE Loss (Real)", "blue"),
        ("ce_fake", "CE Loss (Fake)", "orange"),
        ("discrepancy", "Discrepancy Loss", "red"),
        ("robustness", "Robustness Loss", "green"),
    ]

    for ax, (key, title, color) in zip(axes.flat, components):
        ax.plot(epochs, history[key], color=color, marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_components.png"), dpi=150)
    plt.close()
    print(f"[Saved] loss_components.png")


def save_metrics_json(history: dict, best_acc: float, output_dir: str):
    metrics = {
        "best_val_acc": best_acc,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
        "history": history,
    }
    path = os.path.join(output_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] metrics.json  (best_acc={best_acc:.2f}%)")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build class mapping  (canonical: lowercase with spaces)
    # ------------------------------------------------------------------
    real_root = os.path.join(args.data_root, "train", "Real-Img")
    fake_root = os.path.join(args.data_root, "train", "Fake-Img")
    val_root  = os.path.join(args.data_root, "val")

    # Use real_root as source of truth for class names
    raw_classes = sorted([
        d for d in os.listdir(real_root)
        if os.path.isdir(os.path.join(real_root, d))
    ])
    # class_to_idx keyed by original name (may have underscores)
    class_to_idx = {cls: i for i, cls in enumerate(raw_classes)}
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}  →  {raw_classes[:5]} ...")

    # ------------------------------------------------------------------
    # 2. Load CLIP + LoRA
    #
    # Diagram (Figure 1):
    #   - CLIP visual backbone  →  Frozen  ❄️
    #   - LoRA adapters on visual backbone  →  Learnable  🔥
    #   - Linear classifier head  →  Learnable  🔥
    #   - Text encoder (unused at runtime)  →  Frozen  ❄️
    # ------------------------------------------------------------------
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k"
    )

    # Step 1: Freeze ALL parameters first (backbone + text encoder)
    for p in clip_model.parameters():
        p.requires_grad = False

    # Step 2: Wrap ONLY the visual encoder with LoRA
    #   get_peft_model injects trainable LoRA matrices into q/k/v/out projections
    #   and explicitly sets requires_grad=True for those LoRA weights only
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
    )
    clip_model.visual = get_peft_model(clip_model.visual, lora_cfg)

    # Step 3: Explicitly ensure LoRA weights are trainable
    #   (safety: some PEFT versions need this after freeze-then-wrap)
    for name, p in clip_model.visual.named_parameters():
        if "lora_" in name:
            p.requires_grad = True

    clip_model = clip_model.to(device)

    model = CLIPClassifier(clip_model, feat_dim=512, num_classes=num_classes).to(device)

    # Step 4: Verify parameter groups — print summary
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    print(f"\n[Model] Parameter summary:")
    print(f"  Total      : {total_params:,}")
    print(f"  Trainable  : {trainable_params:,}  (LoRA adapters + linear head)")
    print(f"  Frozen     : {frozen_params:,}  (CLIP visual backbone + text encoder)")
    print(f"  Trainable  : {100.*trainable_params/total_params:.2f}% of total\n")

    # Sanity check: no backbone weight should be trainable
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert ("lora_" in name or "head." in name), (
                f"[ERROR] Unexpected trainable param: {name}. "
                "Only LoRA weights and linear head should be trainable."
            )

    # ------------------------------------------------------------------
    # 3. Datasets & loaders
    # ------------------------------------------------------------------
    real_ds = FolderDataset(real_root, class_to_idx, preprocess, "real")
    fake_ds = FolderDataset(fake_root, class_to_idx, preprocess, "fake")
    val_ds  = FolderDataset(val_root,  class_to_idx, preprocess, "val")

    # pin_memory only helps when CUDA is available
    pin = torch.cuda.is_available()

    # num_workers > 0 can cause issues on Windows; force 0 automatically
    nw = args.num_workers if os.name != "nt" else 0
    if os.name == "nt" and args.num_workers > 0:
        print("[INFO] Windows detected: setting num_workers=0 to avoid DataLoader multiprocessing issues.")

    # Shuffle=False so index i always refers to the same sample
    real_loader = DataLoader(real_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=pin)
    fake_loader = DataLoader(fake_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=pin)
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=pin)

    # Training loaders with shuffle=True for CE loss passes
    real_train_loader = DataLoader(real_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=nw, pin_memory=pin)
    fake_train_loader = DataLoader(fake_ds, batch_size=args.batch_size,
                                   shuffle=True, num_workers=nw, pin_memory=pin)

    # ------------------------------------------------------------------
    # 4. Phase 1 — Fixed K-means clustering  (paper Section 4.1.1)
    #    "we decided to perform clustering on data space to avoid
    #     recomputing the clustering at each iteration"
    # ------------------------------------------------------------------
    print("\n[Phase 1] Extracting features for K-means clustering ...")
    real_feat_np = extract_features(model, real_loader, device)
    fake_feat_np = extract_features(model, fake_loader, device)

    all_feat_np = np.concatenate([real_feat_np, fake_feat_np], axis=0)
    n_clusters = num_classes * 2   # paper default: 2x num_classes
    print(f"  Running K-means with {n_clusters} clusters on {len(all_feat_np)} samples ...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20, max_iter=300)
    all_cluster_ids = kmeans.fit_predict(all_feat_np)

    # Save KMeans model + metadata so it can be shared / reused
    import joblib, json as _json
    kmeans_path = os.path.join(args.output_dir, "kmeans.pkl")
    joblib.dump(kmeans, kmeans_path)

    kmeans_meta = {
        "n_clusters":       n_clusters,
        "n_real_samples":   len(real_ds),
        "n_fake_samples":   len(fake_ds),
        "real_cluster_ids": all_cluster_ids[:len(real_ds)].tolist(),
        "fake_cluster_ids": all_cluster_ids[len(real_ds):].tolist(),
        "seed":             args.seed,
        "clip_model":       "ViT-B-16 laion2b_s34b_b88k",
        "classes":          raw_classes,
    }
    kmeans_meta_path = os.path.join(args.output_dir, "kmeans_meta.json")
    with open(kmeans_meta_path, "w") as _f:
        _json.dump(kmeans_meta, _f, indent=2)

    print(f"  [Saved] kmeans.pkl        → {kmeans_path}")
    print(f"  [Saved] kmeans_meta.json  → {kmeans_meta_path}")

    # Split cluster assignments back
    real_cluster_ids = all_cluster_ids[:len(real_ds)]   # shape (N_real,)
    fake_cluster_ids = all_cluster_ids[len(real_ds):]   # shape (N_fake,)

    # Pre-compute per-cluster membership lists (sample indices)
    real_by_cluster: dict[int, list] = defaultdict(list)
    fake_by_cluster: dict[int, list] = defaultdict(list)
    for idx, cid in enumerate(real_cluster_ids):
        real_by_cluster[int(cid)].append(idx)
    for idx, cid in enumerate(fake_cluster_ids):
        fake_by_cluster[int(cid)].append(idx)

    # Pre-compute per-cluster real counts for robustness weighting
    real_count_by_cluster = {cid: len(idxs) for cid, idxs in real_by_cluster.items()}
    total_real = len(real_ds)
    total_fake = len(fake_ds)

    print(f"  Clusters with both real & fake: "
          f"{sum(1 for c in real_by_cluster if c in fake_by_cluster)}/{n_clusters}")

    # ------------------------------------------------------------------
    # 5. Optimizer + LR Scheduler
    #
    # Two param groups with different LR:
    #   - LoRA adapters : lr        (learnable, needs larger lr)
    #   - Linear head   : lr * 10   (fresh random init, needs even larger lr)
    # Scheduler: CosineAnnealingLR — smoothly decays lr to lr_min
    # Warmup: LinearLR for first `warmup_epochs` epochs
    # ------------------------------------------------------------------
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    head_params  = [p for n, p in model.named_parameters()
                   if p.requires_grad and "head." in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr,        "name": "lora"},
            {"params": head_params,  "lr": args.lr * 10.0, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )

    # Warmup for first 5 epochs, then cosine decay to lr/100
    warmup_epochs   = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - warmup_epochs, 1),
        eta_min=args.lr / 100.0,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------------
    # 6. Phase 2 — Training loop  (Algorithm 2 / Equation 7)
    #
    # Correct mini-batch implementation:
    #   Each epoch = one pass over real + one pass over fake (shuffled)
    #   For each mini-batch:
    #     - CE loss computed with grad (real + fake)
    #     - Discrepancy & robustness computed WITH GRAD using
    #       per-cluster features re-forwarded inside the batch step
    #       → gradient flows through ALL loss terms
    # ------------------------------------------------------------------
    history = {
        "loss": [], "ce_real": [], "ce_fake": [],
        "discrepancy": [], "robustness": [], "val_acc": [], "lr": [],
    }
    best_acc = 0.0

    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.0)

    # Build index-to-sample lookup tensors for cluster reg losses
    # We store raw images per cluster so we can re-forward with grad
    # Use a lighter approach: store dataset indices, re-load per cluster
    # sample a fixed number of representatives per cluster per batch
    N_REP = 8   # max samples per cluster side for reg loss (memory-efficient)

    # Pre-build cluster index lists (already done above, reuse)
    # real_by_cluster, fake_by_cluster  →  dict[cid, list[dataset_idx]]

    # Use torch amp new API to avoid FutureWarning
    use_amp = (device == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"\n[Phase 2] Training for {args.epochs} epochs "
          f"(early stop patience={args.patience}) ...")

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_ce_r = 0.0
        epoch_ce_f = 0.0
        epoch_dis  = 0.0
        epoch_rob  = 0.0
        n_steps    = 0

        # Zip real and fake loaders; cycle the shorter one
        real_iter = iter(real_train_loader)
        fake_iter = iter(fake_train_loader)
        n_steps_per_epoch = max(len(real_train_loader), len(fake_train_loader))

        for step in range(n_steps_per_epoch):

            # --- fetch batches (cycle if exhausted) ---
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter  = iter(real_train_loader)
                real_batch = next(real_iter)

            try:
                fake_batch = next(fake_iter)
            except StopIteration:
                fake_iter  = iter(fake_train_loader)
                fake_batch = next(fake_iter)

            real_imgs   = real_batch["image"].to(device)
            real_labels = real_batch["label"].to(device)
            real_idxs   = real_batch["idx"]          # dataset indices

            fake_imgs   = fake_batch["image"].to(device)
            fake_labels = fake_batch["label"].to(device)
            fake_idxs   = fake_batch["idx"]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):

                # ── CE real (CutMix / Mixup / plain) ──────────────────
                use_mix = random.random()
                if use_mix < 0.5:
                    imgs_r, la, lb, lam = apply_cutmix(real_imgs, real_labels)
                    feat_r, logits_r = model(imgs_r)
                    ce_real = mixup_criterion(criterion, logits_r, la, lb, lam)
                elif use_mix < 0.75:
                    imgs_r, la, lb, lam = apply_mixup(real_imgs, real_labels)
                    feat_r, logits_r = model(imgs_r)
                    ce_real = mixup_criterion(criterion, logits_r, la, lb, lam)
                else:
                    feat_r, logits_r = model(real_imgs)
                    ce_real = criterion(logits_r, real_labels)

                # re-encode without aug for reg loss (need clean features)
                with torch.no_grad():
                    feat_r_clean = model.encode(real_imgs)

                # ── CE fake (CutMix / Mixup / plain) ──────────────────
                use_mix = random.random()
                if use_mix < 0.5:
                    imgs_f, la, lb, lam = apply_cutmix(fake_imgs, fake_labels)
                    feat_f, logits_f = model(imgs_f)
                    ce_fake = mixup_criterion(criterion, logits_f, la, lb, lam)
                elif use_mix < 0.75:
                    imgs_f, la, lb, lam = apply_mixup(fake_imgs, fake_labels)
                    feat_f, logits_f = model(imgs_f)
                    ce_fake = mixup_criterion(criterion, logits_f, la, lb, lam)
                else:
                    feat_f, logits_f = model(fake_imgs)
                    ce_fake = criterion(logits_f, fake_labels)

                # re-encode without aug for reg loss
                feat_f_clean, _ = model(fake_imgs)

                # ── Discrepancy loss (A1) ──────────────────────────────
                # For each cluster represented in this batch:
                #   d = ||mean(real_feat in cluster) - mean(fake_feat in cluster)||
                dis = torch.tensor(0.0, device=device)
                n_dis_clusters = 0

                # find which clusters are present in this batch
                batch_real_cids = set(int(real_cluster_ids[i]) for i in real_idxs.tolist())
                batch_fake_cids = set(int(fake_cluster_ids[i]) for i in fake_idxs.tolist())
                common_cids     = batch_real_cids & batch_fake_cids

                for cid in common_cids:
                    # mask samples belonging to this cluster in current batch
                    r_mask = torch.tensor(
                        [real_cluster_ids[i] == cid for i in real_idxs.tolist()],
                        dtype=torch.bool, device=device)
                    f_mask = torch.tensor(
                        [fake_cluster_ids[i] == cid for i in fake_idxs.tolist()],
                        dtype=torch.bool, device=device)

                    if r_mask.sum() == 0 or f_mask.sum() == 0:
                        continue

                    r_center = feat_r_clean[r_mask].mean(0)
                    f_center = feat_f_clean[f_mask].mean(0)
                    weight   = f_mask.sum().float() / total_fake
                    dis      = dis + weight * torch.norm(r_center - f_center, p=2)
                    n_dis_clusters += 1

                # ── Robustness loss (A2) ───────────────────────────────
                # For each cluster in fake batch: pairwise distance among
                # fake samples within the cluster, weighted by n_real_i/n
                rob = torch.tensor(0.0, device=device)

                for cid in batch_fake_cids:
                    f_mask = torch.tensor(
                        [fake_cluster_ids[i] == cid for i in fake_idxs.tolist()],
                        dtype=torch.bool, device=device)

                    if f_mask.sum() < 2:
                        continue

                    f_feat_cid = feat_f_clean[f_mask]
                    dist       = torch.cdist(f_feat_cid, f_feat_cid, p=2)
                    rob_val    = dist.mean()
                    weight     = real_count_by_cluster.get(cid, 0) / max(total_real, 1)
                    rob        = rob + weight * rob_val

                # ── Total loss (Equation 7) ────────────────────────────
                loss = (
                    args.lambda_real * ce_real
                    + ce_fake
                    + args.lambda_dis * dis
                    + args.lambda_rob * rob
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_ce_r += ce_real.item()
            epoch_ce_f += ce_fake.item()
            epoch_dis  += dis.item()
            epoch_rob  += rob.item()
            n_steps    += 1

        # ── End of epoch: average metrics ──────────────────────────────
        avg_loss = epoch_loss / n_steps
        avg_ce_r = epoch_ce_r / n_steps
        avg_ce_f = epoch_ce_f / n_steps
        avg_dis  = epoch_dis  / n_steps
        avg_rob  = epoch_rob  / n_steps

        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        # Log current LR
        current_lr = optimizer.param_groups[0]["lr"]

        history["loss"].append(avg_loss)
        history["ce_real"].append(avg_ce_r)
        history["ce_fake"].append(avg_ce_f)
        history["discrepancy"].append(avg_dis)
        history["robustness"].append(avg_rob)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch+1:3d}/{args.epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"ce_r={avg_ce_r:.4f}  "
            f"ce_f={avg_ce_f:.4f}  "
            f"dis={avg_dis:.4f}  "
            f"rob={avg_rob:.4f}  "
            f"val_acc={val_acc:.2f}%  "
            f"lr={current_lr:.2e}"
        )

        # ── Save best model ─────────────────────────────────────────────
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best_model.pth"),
            )
            print(f"  → Best model saved (acc={best_acc:.2f}%)")

        # ── Early stopping ──────────────────────────────────────────────
        if early_stopping.step(val_acc):
            print(f"\n[Early Stop] No improvement for {args.patience} epochs. "
                  f"Stopping at epoch {epoch+1}.")
            break

        # ------------------------------------------------------------------
    # 7. Save final metrics & plots
    # ------------------------------------------------------------------
    print("\n[Saving results ...]")
    plot_metrics(history, args.output_dir)
    plot_loss_components(history, args.output_dir)
    save_metrics_json(history, best_acc, args.output_dir)

    # Also save last checkpoint
    torch.save(
        {
            "epoch": len(history["val_acc"]),
            "model_state_dict": model.state_dict(),
            "val_acc": history["val_acc"][-1],
            "args": vars(args),
        },
        os.path.join(args.output_dir, "last_model.pth"),
    )
    print(f"[Saved] last_model.pth")
    print(f"\n{'='*50}")
    print(f"Training complete.  Best Val Acc: {best_acc:.2f}%")
    print(f"Outputs saved to: {args.output_dir}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Few-shot CLIP fine-tuning with synthetic data (ICML 2025)"
    )
    parser.add_argument("--data_root",   default="dataset",  help="Root of dataset folder")
    parser.add_argument("--output_dir",  default="outputs",  help="Where to save outputs")

    # Training
    parser.add_argument("--epochs",      type=int,   default=150,  help="Training epochs (paper: 150 for lightweight)")
    parser.add_argument("--batch_size",  type=int,   default=128,   help="Batch size per pass")
    parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay",type=float, default=1e-4, help="Weight decay")

    # Loss weights  (paper: λ=4, λ1=0.1, λ2=1 for lightweight)
    parser.add_argument("--lambda_real", type=float, default=4.0,  help="Weight for real CE loss")
    parser.add_argument("--lambda_dis",  type=float, default=0.1,  help="Weight for discrepancy loss")
    parser.add_argument("--lambda_rob",  type=float, default=1.0,  help="Weight for robustness loss")

    parser.add_argument("--num_workers", type=int,   default=8)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--patience",    type=int,   default=15,
                        help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
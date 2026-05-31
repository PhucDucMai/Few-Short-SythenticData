"""
Few-Shot Image Classification with Synthetic Data
Based on: "Provably Improving Generalization of Few-Shot Models with Synthetic Data" (ICML 2025)

Loss (Eq. 7):
    L = lambda * F(S,h) + F(G,h) + lambda_1 * Loss_disc - lambda_2 * Loss_rob

Usage examples:
    # Minimal — chỉ cần chỉ đường dẫn dataset
    python train_fewshot.py --data_root ./Datasets

    # Đầy đủ
    python train_fewshot.py \
        --data_root     ./Datasets \
        --output_dir    ./trained_CLIP_V4 \
        --epochs        50 \
        --batch_size    32 \
        --lr            1e-4 \
        --lam           4.0 \
        --lam1          0.1 \
        --lam2          1.0 \
        --n_clusters    0 \
        --lora_rank     4 \
        --seed          0

Requirements:
    pip install torch torchvision open-clip-torch peft scikit-learn tqdm
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    raise ImportError("Run: pip install open-clip-torch")

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    raise ImportError("Run: pip install peft")


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Few-shot CLIP fine-tuning with synthetic data (ICML 2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_root",   type=str, default="./Datasets",
                   help="Root folder containing train/, val/, test/")
    p.add_argument("--output_dir",  type=str, default="./trained_CLIP_V4",
                   help="Directory to save checkpoints and logs")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--clip_model",    type=str, default="ViT-B-16",
                   help="open_clip model name")
    p.add_argument("--clip_pretrain", type=str, default="openai",
                   help="open_clip pretrained weights tag")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument("--lora_rank",    type=int,   default=4)
    p.add_argument("--lora_alpha",   type=int,   default=8)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    # Correct target modules for open_clip ViT (fused attention — no q/v_proj)
    p.add_argument("--lora_targets", type=str,   default="out_proj,c_fc,c_proj",
                   help="Comma-separated LoRA target module names")

    # ── Clustering ────────────────────────────────────────────────────────────
    p.add_argument("--n_clusters",  type=int, default=74,
                   help="K for K-Means. 0 = auto (2 × num_classes)")
    p.add_argument("--kmeans_iter", type=int, default=300,
                   help="Max K-Means iterations")

    # ── Loss hyper-parameters (Eq. 7) ─────────────────────────────────────────
    p.add_argument("--lam",  type=float, default=4.0,
                   help="λ  — weight on real CE loss (use 1.0 for Stanford Cars)")
    p.add_argument("--lam1", type=float, default=0.1,
                   help="λ₁ — weight on discrepancy term")
    p.add_argument("--lam2", type=float, default=1.0,
                   help="λ₂ — weight on robustness term")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-3)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--log_interval",type=int,   default=20,
                   help="Log every N batches")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logging — writes to both stdout and output_dir/train.log
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    logger = logging.getLogger("fewshot")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# CLIP preprocessing
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

feat_transform = transforms.Compose([   # no augmentation, used for K-Means
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """
    Loads train/Real-Img and train/Fake-Img.
    Returns: (image_tensor, class_label: int, is_real: int)
      is_real = 1 for Real-Img, 0 for Fake-Img
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        real_root = Path(root) / "Real-Img"
        fake_root = Path(root) / "Fake-Img"

        if not real_root.exists():
            raise FileNotFoundError(f"Real-Img folder not found: {real_root}")
        if not fake_root.exists():
            raise FileNotFoundError(f"Fake-Img folder not found: {fake_root}")

        self.classes = sorted([d.name for d in real_root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls in self.classes:
            idx = self.class_to_idx[cls]
            for ext in exts:
                for p in (real_root / cls).glob(f"*{ext}"):
                    self.samples.append((str(p), idx, 1))
                for p in (fake_root / cls).glob(f"*{ext}"):
                    self.samples.append((str(p), idx, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_real = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, is_real


class IndexedTrainDataset(Dataset):
    """Wraps TrainDataset and also returns the global sample index."""

    def __init__(self, base: TrainDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label, is_real = self.base[idx]
        return img, label, is_real, idx


class EvalDataset(Dataset):
    """Flat class-folder layout used by val/ and test/."""

    def __init__(self, root: str, class_to_idx: Dict[str, int], transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for cls, idx in class_to_idx.items():
            cls_dir = Path(root) / cls
            if cls_dir.exists():
                for ext in exts:
                    for p in cls_dir.glob(f"*{ext}"):
                        self.samples.append((str(p), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# Model: CLIP ViT-B/16 + LoRA + Linear head
# ─────────────────────────────────────────────────────────────────────────────

class CLIPLoRAClassifier(nn.Module):

    def __init__(self, num_classes: int, args, logger):
        super().__init__()

        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrain
        )
        self.encoder = clip_model.visual

        # Freeze all encoder weights
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Apply LoRA — targets the Linear layers that exist in open_clip ViT
        target_modules = [t.strip() for t in args.lora_targets.split(",")]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)

        # Infer feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.encoder(dummy).shape[-1]

        self.classifier = nn.Linear(feat_dim, num_classes)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Model ready — {trainable:,} trainable / {total:,} total params "
            f"({100 * trainable / total:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability vectors (B × num_classes)."""
        feats  = self.encoder(x)
        logits = self.classifier(feats)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# K-Means clustering (once before training)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    dataset: TrainDataset,
    encoder: nn.Module,
    device: str,
    num_workers: int,
) -> np.ndarray:
    orig_tf = dataset.transform
    dataset.transform = feat_transform

    loader = DataLoader(dataset, batch_size=64, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    encoder.eval()
    feats = []
    with torch.no_grad():
        for imgs, _, _ in tqdm(loader, desc="  Extracting features", leave=False):
            feats.append(encoder(imgs.to(device)).cpu().numpy())

    dataset.transform = orig_tf
    return np.concatenate(feats, axis=0)


def run_kmeans(features: np.ndarray, n_clusters: int,
               max_iter: int, seed: int, logger) -> np.ndarray:
    logger.info(f"Running K-Means  k={n_clusters}  n={len(features)} ...")
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                n_init=10, random_state=seed)
    labels = km.fit_predict(features)
    logger.info(f"K-Means done in {time.time()-t0:.1f}s")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Custom Loss — Equation 7
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(
    probs_real:       torch.Tensor,   # (N_r, C)
    labels_real:      torch.Tensor,   # (N_r,)
    probs_fake:       torch.Tensor,   # (N_f, C)
    labels_fake:      torch.Tensor,   # (N_f,)
    cluster_ids_real: torch.Tensor,   # (N_r,)
    cluster_ids_fake: torch.Tensor,   # (N_f,)
    g_total:          int,
    lam:  float,
    lam1: float,
    lam2: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    eps = 1e-8

    # ── Cross-entropy ──────────────────────────────────────────────────────
    ce_real = F.nll_loss(torch.log(probs_real + eps), labels_real)
    ce_fake = F.nll_loss(torch.log(probs_fake + eps), labels_fake)

    # ── Cluster-wise regularization ────────────────────────────────────────
    T_S = cluster_ids_real.unique().tolist()   # clusters with ≥1 real image

    disc = torch.zeros(1, device=probs_real.device)
    rob  = torch.zeros(1, device=probs_real.device)

    for i in T_S:
        h_s = probs_real[cluster_ids_real == i]   # (|S_i|, C)
        h_g = probs_fake[cluster_ids_fake == i]   # (|G_i|, C)

        if h_g.shape[0] == 0:
            continue

        g_i = h_g.shape[0]
        w   = g_i / g_total                       # scalar weight

        # Discrepancy: mean ||h(s) − h(g)||_2 over all (s,g) pairs in cluster i
        # Shape: (|S_i|, |G_i|, C) → (|S_i|, |G_i|) → scalar
        diff_d = h_s.unsqueeze(1) - h_g.unsqueeze(0)
        disc   = disc + w * torch.norm(diff_d, p=2, dim=-1).mean()

        # Robustness: mean ||h(g1) − h(g2)||_2 over all fake pairs in cluster i
        if g_i < 2:
            continue
        diff_r     = h_g.unsqueeze(1) - h_g.unsqueeze(0)   # (G_i, G_i, C)
        pw_r       = torch.norm(diff_r, p=2, dim=-1)        # (G_i, G_i)
        off_diag   = ~torch.eye(g_i, dtype=torch.bool, device=h_g.device)
        rob        = rob + (1.0 / g_total) * (pw_r[off_diag].sum() / g_i)

    # ── Final loss (Eq. 7) ─────────────────────────────────────────────────
    loss = lam * ce_real + ce_fake + lam1 * disc - lam2 * rob

    return loss, {
        "ce_real": ce_real.item(),
        "ce_fake": ce_fake.item(),
        "disc":    disc.item(),
        "rob":     rob.item(),
        "total":   loss.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device : {device}")
    logger.info(f"Args   : {vars(args)}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_base = TrainDataset(
        os.path.join(args.data_root, "train"), transform=train_transform
    )
    num_classes = len(train_base.classes)

    n_real = sum(1 for *_, r in train_base.samples if r)
    n_fake = sum(1 for *_, r in train_base.samples if not r)
    logger.info(f"Classes: {num_classes}  |  Real: {n_real}  |  Fake: {n_fake}")

    val_ds  = EvalDataset(
        os.path.join(args.data_root, "val"),
        train_base.class_to_idx, eval_transform
    )
    test_ds = EvalDataset(
        os.path.join(args.data_root, "test"),
        train_base.class_to_idx, eval_transform
    )

    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CLIPLoRAClassifier(num_classes, args, logger).to(device)

    # ── K-Means (once) ────────────────────────────────────────────────────────
    n_clusters = args.n_clusters if args.n_clusters > 0 else 2 * num_classes
    logger.info(f"Clusters: {n_clusters}")

    logger.info("Phase 1 — feature extraction for K-Means")
    features   = extract_features(train_base, model.encoder, device, args.num_workers)
    cluster_ids = run_kmeans(features, n_clusters, args.kmeans_iter, args.seed, logger)
    cluster_ids_tensor = torch.tensor(cluster_ids, dtype=torch.long)

    g_total = n_fake
    logger.info(f"g_total (total synthetic images): {g_total}")

    # ── DataLoader for training ───────────────────────────────────────────────
    train_loader = DataLoader(
        IndexedTrainDataset(train_base),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Phase 2 — training")
    best_val_acc   = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        accum = {"ce_real": 0.0, "ce_fake": 0.0,
                 "disc": 0.0, "rob": 0.0, "total": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)

        for b_idx, (imgs, labels, is_real_flags, sample_idxs) in enumerate(pbar):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            real_mask = is_real_flags.bool()
            fake_mask = ~real_mask

            if real_mask.sum() == 0 or fake_mask.sum() == 0:
                continue    # skip degenerate batch

            probs_all = model(imgs)

            probs_real  = probs_all[real_mask]
            labels_real = labels[real_mask]
            probs_fake  = probs_all[fake_mask]
            labels_fake = labels[fake_mask]

            c_real = cluster_ids_tensor[sample_idxs[real_mask]].to(device)
            c_fake = cluster_ids_tensor[sample_idxs[fake_mask]].to(device)

            loss, info = compute_loss(
                probs_real, labels_real,
                probs_fake, labels_fake,
                c_real, c_fake,
                g_total,
                lam=args.lam, lam1=args.lam1, lam2=args.lam2,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in accum:
                accum[k] += info[k if k != "total" else "total"]
            n_batches += 1

            if (b_idx + 1) % args.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{info['total']:.4f}",
                    "disc": f"{info['disc']:.5f}",
                    "rob":  f"{info['rob']:.5f}",
                })

        scheduler.step()

        if n_batches == 0:
            continue

        avg = {k: v / n_batches for k, v in accum.items()}
        val_acc = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg['total']:.4f} | "
            f"ce_real={avg['ce_real']:.4f} | "
            f"ce_fake={avg['ce_fake']:.4f} | "
            f"disc={avg['disc']:.5f} | "
            f"rob={avg['rob']:.5f} | "
            f"val_acc={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "val_acc":    val_acc,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "args":       vars(args),
            }, best_ckpt_path)
            logger.info(f"  ✓ Best val acc {best_val_acc:.2f}% — saved to {best_ckpt_path}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint: {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_acc = evaluate(model, test_loader, device)

    logger.info("=" * 60)
    logger.info(f"Best val acc : {best_val_acc:.2f}%")
    logger.info(f"Test acc     : {test_acc:.2f}%")
    logger.info("=" * 60)

    # Save final results summary
    summary_path = os.path.join(args.output_dir, "results.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"best_val_acc : {best_val_acc:.2f}%\n")
        f.write(f"test_acc     : {test_acc:.2f}%\n")
        f.write(f"args         : {vars(args)}\n")
    logger.info(f"Results saved to {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(parse_args())
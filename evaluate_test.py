from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from peft import LoraConfig, get_peft_model


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "model" / "best_model.pth"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MODEL_INPUT_SIZE = 224


class CLIPLoRAClassifier(nn.Module):
    def __init__(self, num_classes: int, args):
        super().__init__()

        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=None
        )
        self.encoder = clip_model.visual

        for param in self.encoder.parameters():
            param.requires_grad_(False)

        target_modules = [target.strip() for target in args.lora_targets.split(",")]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, lora_cfg)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
            feat_dim = self.encoder(dummy).shape[-1]

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return F.softmax(logits, dim=-1)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test split")
    parser.add_argument("--data_root", type=str, default=str(ROOT), help="Root containing test/")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "test_eval"))
    return parser.parse_args()


def resolve_device(choice: str) -> str:
    if choice == "cpu":
        return "cpu"
    if choice == "gpu":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]
    num_classes = state_dict["classifier.weight"].shape[0]
    args = SimpleNamespace(**checkpoint.get("args", {}))

    model = CLIPLoRAClassifier(num_classes=num_classes, args=args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        probs = model(images)
        preds = probs.argmax(dim=1).cpu().numpy()

        all_targets.extend(targets.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.max(dim=1).values.cpu().numpy().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    report_text = classification_report(all_targets, all_preds, target_names=loader.dataset.classes, zero_division=0)
    report_dict = classification_report(
        all_targets,
        all_preds,
        target_names=loader.dataset.classes,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_targets, all_preds)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mean_confidence": float(np.mean(all_probs)) if all_probs else 0.0,
        "report": report_text,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "targets": all_targets,
        "preds": all_preds,
    }


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Set Confusion Matrix")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    test_dir = data_root / "test"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    dataset = datasets.ImageFolder(test_dir, transform=build_transform())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model, checkpoint, num_classes = load_model(checkpoint_path, device)

    if len(dataset.classes) != num_classes:
        print(f"[WARN] Model has {num_classes} classes but test folder has {len(dataset.classes)} folders.")

    metrics = evaluate(model, loader, device)
    cm = metrics["confusion_matrix"]

    print("=" * 72)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device}")
    print(f"Test split : {test_dir}")
    print(f"Samples    : {len(dataset)}")
    print(f"Classes    : {len(dataset.classes)}")
    print(f"Accuracy   : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1   : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%")
    print(f"Mean conf. : {metrics['mean_confidence'] * 100:.2f}%")
    print("=" * 72)
    print(metrics["report"])

    class_names = dataset.classes
    save_confusion_matrix(cm, class_names, output_dir / "test_confusion_matrix.png")

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": device,
        "test_dir": str(test_dir),
        "num_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mean_confidence": metrics["mean_confidence"],
        "class_names": class_names,
        "report_dict": metrics["report_dict"],
        "val_acc_from_checkpoint": checkpoint.get("val_acc"),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved confusion matrix to: {output_dir / 'test_confusion_matrix.png'}")
    print(f"Saved metrics to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
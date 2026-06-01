from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import open_clip
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from streamlit.runtime.scriptrunner import get_script_run_ctx


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "model" / "best_model.pth"
DATASET_INFO_PATH = ROOT / "Visualization-Image" / "dataset_info.json"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MODEL_INPUT_SIZE = 224
DISPLAY_SIZE = (224, 224)

DEFAULT_CLASS_NAMES = [
    "Abyssinian",
    "American_Bulldog",
    "American_Pit_Bull_Terrier",
    "Basset_Hound",
    "Beagle",
    "Bengal",
    "Birman",
    "Bombay",
    "Boxer",
    "British_Shorthair",
    "Chihuahua",
    "Egyptian_Mau",
    "English_Cocker_Spaniel",
    "English_Setter",
    "German_Shorthaired",
    "Great_Pyrenees",
    "Havanese",
    "Japanese_Chin",
    "Keeshond",
    "Leonberger",
    "Main_Coon",
    "Miniature_Pinscher",
    "Newfoundland",
    "Persian",
    "Pomeranian",
    "Pug",
    "Ragdoll",
    "Russian_Blue",
    "Saint_Bernard",
    "Samoyed",
    "Scottish_Terrier",
    "Shiba_Inu",
    "Siamese",
    "Sphynx",
    "Staffordshire_Bull_Terrier",
    "Wheaten_Terrier",
    "Yorkshire_Terrier",
]


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


def _load_dataset_class_names(num_classes: int) -> Sequence[str]:
    if DATASET_INFO_PATH.exists():
        try:
            payload = json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
            class_names = payload.get("classes", [])
            if isinstance(class_names, list) and len(class_names) == num_classes:
                return class_names
        except Exception:
            pass

    class_names = sorted(DEFAULT_CLASS_NAMES)
    if len(class_names) == num_classes:
        return class_names

    return [f"class_{idx}" for idx in range(num_classes)]


def _build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(MODEL_INPUT_SIZE),
            transforms.CenterCrop(MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def _prepare_display_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), DISPLAY_SIZE, method=Image.Resampling.LANCZOS)


@st.cache_resource(show_spinner=False)
def load_model(device_preference: str):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    if device_preference == "gpu" and not torch.cuda.is_available():
        device = "cpu"
    elif device_preference == "gpu":
        device = "cuda"
    else:
        device = "cpu"

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["model"]
    num_classes = state_dict["classifier.weight"].shape[0]
    args = SimpleNamespace(**checkpoint.get("args", {}))

    model = CLIPLoRAClassifier(num_classes=num_classes, args=args).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    class_names = _load_dataset_class_names(num_classes)
    preprocess = _build_preprocess()
    return model, preprocess, class_names, device, checkpoint


def predict(image: Image.Image, device_preference: str):
    model, preprocess, class_names, device, _ = load_model(device_preference)
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model(tensor)[0].detach().cpu()

    top_prob, top_idx = torch.max(probs, dim=0)
    top_label = class_names[int(top_idx)]

    top_k = min(3, len(class_names))
    top_values, top_indices = torch.topk(probs, k=top_k)
    ranked = [
        {"class": class_names[int(idx)], "probability": float(value)}
        for value, idx in zip(top_values, top_indices)
    ]
    return top_label, float(top_prob), ranked


def main() -> None:
    if get_script_run_ctx() is None:
        print("This app must be launched with: streamlit run streamlit_app.py")
        return

    st.set_page_config(
        page_title="Pet Class Demo",
        page_icon="🐾",
        layout="wide",
    )

    st.title("Demo model phân loại ảnh pet")
    st.write(
        "Upload một ảnh, hệ thống sẽ resize về 512x512 để xem trước và đưa qua model để dự đoán class."
    )

    device_options = ["cpu"]
    if torch.cuda.is_available():
        device_options.insert(0, "gpu")

    with st.sidebar:
        st.subheader("Inference")
        device_preference = st.radio(
            "Chọn thiết bị",
            options=device_options,
            format_func=lambda value: "GPU" if value == "gpu" else "CPU",
            index=0 if device_options[0] == "gpu" else 0,
        )
        if device_preference == "gpu" and not torch.cuda.is_available():
            st.warning("GPU không khả dụng trong môi trường hiện tại, app sẽ dùng CPU.")

    try:
        _, _, class_names, device, checkpoint = load_model(device_preference)
    except Exception as exc:
        st.error(f"Không thể tải checkpoint: {exc}")
        st.stop()

    with st.sidebar:
        st.subheader("Model info")
        st.write(f"Checkpoint: {CHECKPOINT_PATH.name}")
        st.write(f"Device: {device}")
        st.write(f"Classes: {len(class_names)}")
        val_acc = checkpoint.get("val_acc", "N/A")
        if isinstance(val_acc, (int, float)):
            st.write(f"Best val acc: {val_acc:.2f}%")
        else:
            st.write(f"Best val acc: {val_acc}")

    uploaded_file = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Hãy upload một ảnh để bắt đầu.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Ảnh không hợp lệ: {exc}")
        return

    resized_image = _prepare_display_image(image)
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Ảnh upload")
        st.image(resized_image, caption="Ảnh đã resize 512x512", use_container_width=True)

    with right:
        st.subheader("Kết quả dự đoán")
        with st.spinner("Đang chạy inference..."):
            label, probability, ranked = predict(resized_image, device_preference)

        st.metric("Class dự đoán", label)
        st.metric("Độ tin cậy", f"{probability * 100:.2f}%")
        st.write("Top 3 dự đoán")
        st.table(
            [
                {
                    "Class": item["class"],
                    "Probability": f"{item['probability'] * 100:.2f}%",
                }
                for item in ranked
            ]
        )


if __name__ == "__main__":
    main()
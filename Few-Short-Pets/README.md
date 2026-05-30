# 🐾 CLIP ViT-B/16 — Pet Breed Classification
## 37-Class Dog & Cat Breed Recognition Pipeline

---

## 📁 Cấu trúc dữ liệu yêu cầu

```
data/
├── Abyssinian/              ← tên folder = tên class
│   ├── Abyssinian_001.jpg
│   ├── Abyssinian_002.jpg
│   └── ...  (~150 ảnh)
├── american_bulldog/
│   ├── american_bulldog_001.jpg
│   └── ...
├── Bengal/
├── Birman/
├── Bombay/
├── British_Shorthair/
├── chihuahua/
├── Egyptian_Mau/
├── english_cocker_spaniel/
├── english_setter/
├── german_shorthaired/
├── great_pyrenean_mountain/
├── havanese/
├── japanese_chin/
├── keeshond/
├── leonberger/
├── Maine_Coon/
├── miniature_pinscher/
├── newfoundland/
├── Persian/
├── pomeranian/
├── pug/
├── Ragdoll/
├── Russian_Blue/
├── saint_bernard/
├── samoyed/
├── scottish_terrier/
├── shiba_inu/
├── Siamese/
├── Sphynx/
├── staffordshire_bull_terrier/
├── wheaten_terrier/
├── yorkshire_terrier/
└── ... (37 classes total)
```

> **Quy tắc tên folder:** Tên folder sẽ tự động trở thành label. Không cần file CSV hay mapping file nào cả.

---

## 📊 Chiến lược phân chia Train / Val / Test

| Split | Tỷ lệ | ~Số ảnh/class | Mục đích |
|-------|--------|---------------|----------|
| **Train** | 75% | ~112 | Cập nhật gradient, học features |
| **Val**   | 15% | ~23  | Monitor overfitting, lựa chọn best model |
| **Test**  | 10% | ~15  | Đánh giá cuối cùng (chỉ chạy 1 lần) |

**Cách thực hiện:** Stratified split theo từng class → đảm bảo phân phối đồng đều.  
Seed cố định `42` → reproducible. **Không copy/move file**, chỉ lưu index.

---

## 🏗️ Kiến trúc mô hình

```
Input (224×224 RGB)
      ↓
CLIP ViT-B/16 Visual Encoder    ← Frozen (epochs 1–9), Fine-tuned (epoch 10+)
      ↓
Image Embeddings [B, 512]
      ↓
LayerNorm → Dropout(0.1) → Linear(512→256) → GELU → Dropout(0.05) → Linear(256→37)
      ↓
Logits [B, 37]
```

**Hai giai đoạn training:**
- **Warm-up (epoch 1–9):** Chỉ train Classification Head. CLIP backbone frozen.
- **Fine-tuning (epoch 10+):** Mở 4 transformer blocks cuối + LN + projection. LR backbone = LR_head × 0.1.

---

## ⚙️ Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Cài dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

---

## 🚀 Chạy pipeline

### Bước 1: Kiểm tra & EDA dữ liệu
```bash
python prepare_data.py \
    --data_root ./data \
    --val_ratio 0.15 \
    --test_ratio 0.10 \
    --sample_sizes
```

### Bước 2: Training
```bash
python train.py \
    --data_root      ./data \
    --output_dir     ./runs/clip_vit_b16 \
    --epochs         30 \
    --batch_size     32 \
    --lr             1e-3 \
    --wd             1e-4 \
    --dropout        0.1 \
    --val_ratio      0.15 \
    --test_ratio     0.10 \
    --unfreeze_epoch 10 \
    --unfreeze_layers 4 \
    --num_workers    4 \
    --device         cuda
```

### Resume từ checkpoint
```bash
python train.py \
    --data_root ./data \
    --resume ./runs/clip_vit_b16/checkpoints/checkpoint_epoch_010.pth
```

---

## 📂 Output structure

```
runs/clip_vit_b16/
├── checkpoints/
│   ├── checkpoint_epoch_005.pth
│   ├── checkpoint_epoch_010.pth
│   ├── ...
│   └── best_model.pth            ← Best val accuracy checkpoint
├── logs/
│   └── training_20241201_143022.log
└── metrics_history.json

Visualization-Image/
├── eda_images_per_class.png
├── eda_split_distribution.png
├── eda_image_sizes.png
├── eda_class_count_histogram.png
├── class_distribution.png
├── learning_curves.png            ← Updated every epoch
├── lr_schedule.png
├── confusion_matrix_epoch_005.png
├── confusion_matrix_epoch_010.png
├── confusion_matrix_epoch_000.png ← Final test evaluation
├── per_class_f1_epoch_XXX.png
├── top_bottom_classes_epoch_XXX.png
└── training_summary.png
```

---

## 🔧 Hyperparameters

| Parameter | Value | Ghi chú |
|-----------|-------|---------|
| Model | ViT-B/16 | CLIP pretrained |
| Input size | 224×224 | CLIP standard |
| Epochs | 30 | Có thể tăng |
| Batch size | 32 | Với GPU 8GB+ |
| LR (head) | 1e-3 | AdamW |
| LR (backbone) | 1e-4 | Sau unfreeze |
| LR scheduler | CosineAnnealing | eta_min = 1e-6 |
| Weight decay | 1e-4 | Regularization |
| Label smoothing | 0.1 | Tránh overconfidence |
| Dropout | 0.1 | Classification head |
| Gradient clipping | 1.0 | Tránh exploding gradient |
| Mixed precision | FP16 | AMP autocast |
| Unfreeze epoch | 10 | Backbone fine-tuning |
| Checkpoint freq | Every 5 epochs | + best model |

---

## 📈 Các biểu đồ được tạo

| Biểu đồ | File | Tần suất |
|---------|------|----------|
| Learning curves (Loss, Top-1, Top-5) | `learning_curves.png` | Mỗi epoch |
| Confusion matrix | `confusion_matrix_epoch_XXX.png` | Mỗi 5 epochs |
| Per-class F1 score | `per_class_f1_epoch_XXX.png` | Mỗi 5 epochs |
| Top/Bottom performing classes | `top_bottom_classes_epoch_XXX.png` | Mỗi 5 epochs |
| LR schedule | `lr_schedule.png` | Mỗi 5 epochs |
| Training summary | `training_summary.png` | Cuối training |
| EDA: Images per class | `eda_images_per_class.png` | Một lần |
| EDA: Split distribution | `eda_split_distribution.png` | Một lần |

---

## 💡 Tips & Tricks

- **GPU < 8GB**: Giảm `--batch_size 16` và dùng `--num_workers 2`
- **CPU only**: Thêm `--device cpu`, sẽ chậm hơn nhiều
- **Tăng performance**: Tăng `--unfreeze_layers 6` hoặc `--unfreeze_epoch 5`
- **Data augmentation**: Có thể mở rộng `train_transform` trong `train.py`
- **Class imbalance**: Nếu các class lệch nhau, thêm `WeightedRandomSampler`

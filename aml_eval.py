import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


# ======================
# Dataset
# ======================
class GlyphCountryDataset(Dataset):
    def __init__(self, root_dir: str, label_to_idx: dict, image_size: int = 64):
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"{self.root_dir} 不存在"

        exts = {".png", ".jpg", ".jpeg", ".bmp"}

        self.image_paths = []
        self.labels = []

        for p in self.root_dir.rglob("*"):
            if p.suffix.lower() in exts:
                family_dir = p.parents[1]
                family_name = family_dir.name

                if "_" in family_name:
                    country = family_name.split("_")[-1]
                else:
                    country = family_name

                if country not in label_to_idx:
                    continue

                self.image_paths.append(p)
                self.labels.append(label_to_idx[country])

        if not self.image_paths:
            raise RuntimeError(f"沒有圖片可供測試")

        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = self.transform(img)
        return img, self.labels[idx]


# ======================
# Model
# ======================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ======================
# Evaluate test set
# ======================
def evaluate_test_set(data_root, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]  # key: int, value: str

    print("Label mapping:", label_to_idx)

    dataset = GlyphCountryDataset(data_root, label_to_idx=label_to_idx)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_idx)

    model = SimpleCNN(num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total = 0
    correct = 0

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

            pbar.set_postfix(acc=f"{correct/total:.4f}")

    print(f"\nFinal accuracy = {correct/total:.4f}")

    print("\n=== 每一類別被誤認成其他類別的比例 ===")
    idx2lbl = {int(k): v for k, v in idx_to_label.items()}

    for true_idx in range(num_classes):
        true_label = idx2lbl[true_idx]
        row = confusion[true_idx]
        total_true = row.sum().item()
        if total_true == 0:
            continue

        print(f"\n[真實標籤 = {true_label}] (總數 {total_true})")
        for pred_idx in range(num_classes):
            pred_label = idx2lbl[pred_idx]
            count = row[pred_idx].item()
            if count == 0:
                continue
            if pred_idx == true_idx:
                correct_rate = count / total_true
                print(f"  ✓ 正確預測為 {pred_label}: {count} 次 "
                      f"({correct_rate:.3f})")
            else:
                mis_rate = count / total_true
                print(f"  ✗ 誤認為 {pred_label}: {count} 次 "
                      f"({mis_rate:.3f})")
    print("\n=== Confusion Matrix (counts) ===")
    print("    " + "  ".join(f"{idx2lbl[j]:>4}" for j in range(num_classes)))
    for i in range(num_classes):
        row_str = "  ".join(f"{confusion[i, j].item():>4}" for j in range(num_classes))
        print(f"{idx2lbl[i]:>4} {row_str}")



# ======================
# Evaluate single image
# ======================
def predict_single_image(img_path, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = ckpt["idx_to_label"]

    transform = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(len(label_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(img.to(device))
        pred = output.argmax(dim=1).item()

    print("Prediction =", idx_to_label[pred])
    return idx_to_label[pred]


# ======================
# Main usage
# ======================
if __name__ == "__main__":
    DATA_ROOT = "/home/t_1/font_data/data"
    CKPT = "checkpoints/font_country_classifier.pth"

    evaluate_test_set(DATA_ROOT, CKPT)

    # predict_single_image("/home/t_1/font_data/data/.../xxx.png", CKPT)

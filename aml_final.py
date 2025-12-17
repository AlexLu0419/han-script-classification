import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


# ======================
# Dataset
# ======================

class GlyphCountryDataset(Dataset):


    def __init__(self, root_dir: str, image_size: int = 64,
                 style: str | None = None,
                 exclude_country: List[str] | None = None):
        self.root_dir = Path(root_dir)
        assert self.root_dir.exists(), f"{self.root_dir} 不存在"

        self.style = style                # e.g. "Bold"
        self.exclude_country = exclude_country or []

        exts = {".png", ".jpg", ".jpeg", ".bmp"}

        self.image_paths = []
        self.label_names = []

        for p in self.root_dir.rglob("*"):
            if p.suffix.lower() in exts:

                style_name = p.parent.name
                if self.style is not None:
                    if not style_name.lower().endswith(self.style.lower()):
                        continue
                family_name = p.parents[1].name  # e.g. Noto_Sans_JP
                if "_" in family_name:
                    country = family_name.split("_")[-1]
                else:
                    country = family_name
                if country in self.exclude_country:
                    continue

                self.image_paths.append(p)
                self.label_names.append(country)

        if not self.image_paths:
            raise RuntimeError(
                f"沒有找到任何圖片（style={self.style}, exclude={self.exclude_country}）"
            )

        unique_labels = sorted(set(self.label_names))
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        print(f"找到 {len(self.image_paths)} 張圖片")
        print(f"國別類別 ({self.num_classes}): {unique_labels}")
        if self.style:
            print(f"字重: {self.style}")
        if self.exclude_country:
            print(f"排除語系: {self.exclude_country}")

        self.transform = T.Compose([
            T.Grayscale(1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.label_names[idx]
        img = Image.open(img_path).convert("L")
        img = self.transform(img)
        return img, self.label_to_idx[label_name]


# ======================
# Model (Simple CNN)
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
# Train / Test loops with tqdm
# ======================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="[Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{(correct / total):.4f}"
        })

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="[Test]", leave=False)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / total):.4f}"
            })

    return running_loss / total, correct / total


# ======================
# Main
# ======================

def main():
    DATA_ROOT = "/home/t_1/font_data/data"
    CKPT_PATH = "checkpoints/font_country_classifier.pth"  

    dataset = GlyphCountryDataset(
        DATA_ROOT,
        image_size=64,
        style=None,           
        # exclude_country=["JP"] 
    )



    # train/test split
    test_ratio = 0.2
    total = len(dataset)
    n_test = int(total * test_ratio)
    n_train = total - n_test

    generator = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)

    model = SimpleCNN(dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    start_epoch = 1
    if os.path.exists(CKPT_PATH):
        print(f"發現 checkpoint：{CKPT_PATH}，載入 model 權重繼續訓練")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("沒有找到舊的 checkpoint，從頭訓練")

    EPOCHS = 50   
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        print(f"\n===== Epoch {epoch} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": dataset.label_to_idx,
            "idx_to_label": dataset.idx_to_label,
        },
        CKPT_PATH,
    )
    print(f"模型已更新儲存到 {CKPT_PATH}")


if __name__ == "__main__":
    main()

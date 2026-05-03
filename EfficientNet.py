import os
import copy
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
import matplotlib.pyplot as plt


class MedicalDataset(Dataset):
    def __init__(self, folders):
        self.samples = []

        valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        for folder_path, class_name in folders:
            label = 0 if class_name == "SJS" else 1
            folder_path = Path(folder_path)

            if not folder_path.exists():
                print(f"Pasta não encontrada: {folder_path}")
                continue

            for img_path in folder_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in valid_ext:
                    self.samples.append((img_path, label, class_name))

        # transforms precisam ficar DENTRO do __init__
        self.resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def augment_sjs(self, img):
        angle = np.random.choice([0, 90, 180, 270])
        img = img.rotate(angle)

        if np.random.rand() > 0.5:
            img = transforms.functional.hflip(img)

        return img

    def augment_cadr(self, img):
        angle = np.random.choice([0, 180])
        img = img.rotate(angle)

        if np.random.rand() > 0.5:
            img = transforms.functional.hflip(img)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if class_name == "SJS":
            img = self.augment_sjs(img)
        else:
            img = self.augment_cadr(img)

        img = self.resize(img)
        img = self.to_tensor(img)
        img = self.normalize(img)

        return img, label


def create_model():
    model = models.efficientnet_b7(weights="DEFAULT")

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    return model


def train_one_fold(
    train_loader,
    val_loader,
    device,
    epochs=50,
    patience=10,
    min_delta=1e-4,
    positive_label=0,
    ema_alpha=0.3,
    warmup_epochs=5
):
    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_sensitivity = -float("inf")
    best_val_loss = float("inf")
    best_model = None
    best_cm = None
    best_auc = None

    patience_counter = 0
    ema_sensitivity = None

    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # validação
        model.eval()
        preds, targets = [], []
        probs_positive = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                probabilities = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)

                preds.extend(pred.cpu().numpy())
                targets.extend(labels.cpu().numpy())

                probs_positive.extend(
                    probabilities[:, positive_label].cpu().numpy()
                )

        val_loss /= len(val_loader.dataset)

        acc = accuracy_score(targets, preds)

        sensitivity = recall_score(
            targets,
            preds,
            pos_label=positive_label,
            zero_division=0
        )

        try:
            auc = roc_auc_score(
                [1 if t == positive_label else 0 for t in targets],
                probs_positive
            )
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(targets, preds)

        if ema_sensitivity is None:
            ema_sensitivity = sensitivity
        else:
            ema_sensitivity = (
                ema_alpha * sensitivity +
                (1 - ema_alpha) * ema_sensitivity
            )

        print(
            f"Epoch {epoch+1}: "
            f"Val Loss = {val_loss:.4f} | "
            f"Acc = {acc:.4f} | "
            f"Sensibilidade = {sensitivity:.4f} | "
            f"EMA Sens = {ema_sensitivity:.4f} | "
            f"AUC = {auc:.4f}"
        )

        improved_sensitivity = ema_sensitivity > best_sensitivity + min_delta
        same_sensitivity_better_loss = (
            abs(ema_sensitivity - best_sensitivity) <= min_delta
            and val_loss < best_val_loss
        )

        if improved_sensitivity or same_sensitivity_better_loss:
            best_sensitivity = ema_sensitivity
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            best_cm = cm.copy()
            best_auc = auc
            patience_counter = 0
        else:
            if epoch + 1 > warmup_epochs:
                patience_counter += 1

        if epoch + 1 > warmup_epochs and patience_counter >= patience:
            print(f"Early stopping na epoch {epoch+1}")
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model, best_sensitivity, best_cm, best_auc


def plot_confusion_matrix(cm, fold):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Matriz de Confusão - Fold {fold}')
    plt.colorbar()

    plt.xticks([0, 1], ['SJS', 'CADR'])
    plt.yticks([0, 1], ['SJS', 'CADR'])

    plt.xlabel("Predito")
    plt.ylabel("Real")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.show()


def run_cross_validation(base_path, batch_size=8, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for val_fold in range(1, 7):
        print(f"\n======== FOLD {val_fold} ========")

        train_folders = []
        val_folders = []

        for fold in range(1, 7):
            sjs_folder = os.path.join(base_path, "SJS", f"pasta{fold}")
            cadr_folder = os.path.join(base_path, "CADR", f"pasta{fold}")

            if fold == val_fold:
                val_folders.append((sjs_folder, "SJS"))
                val_folders.append((cadr_folder, "CADR"))
            else:
                train_folders.append((sjs_folder, "SJS"))
                train_folders.append((cadr_folder, "CADR"))

        train_dataset = MedicalDataset(train_folders)
        val_dataset = MedicalDataset(val_folders)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model, best_sensitivity, best_cm, best_auc = train_one_fold(
            train_loader,
            val_loader,
            device,
            epochs,
            patience=10,
            positive_label=0
        )

        results.append(best_sensitivity)

        print(f"\nMatriz de confusão do Fold {val_fold}:")
        print(f"Melhor sensibilidade do Fold {val_fold}: {best_sensitivity:.4f}")
        print(f"AUC do melhor modelo do Fold {val_fold}: {best_auc:.4f}")
        print(best_cm)


run_cross_validation(r"C:\Users\eduqu\OneDrive\Documentos\GitHub\IC\dataset", batch_size=8, epochs=50)


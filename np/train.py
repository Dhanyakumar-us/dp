import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TORCH_HOME", "D:/torch_cache")

import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

from model import get_dataloaders, TransferClassifier, DEFAULT_CLASSES, CIFAR10_ALL_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transfer Learning Image Classifier on CIFAR Subset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for classifier head")
    parser.add_argument("--samples_train", type=int, default=600, help="Train samples per class")
    parser.add_argument("--samples_val", type=int, default=150, help="Validation samples per class")
    parser.add_argument("--img_size", type=int, default=96, help="Input image dimension (height=width)")
    parser.add_argument("--fine_tune_epoch", type=int, default=3, help="Epoch to unfreeze top layers (0 to disable)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save artifacts")
    parser.add_argument("--samples_dir", type=str, default="sample_images", help="Directory to export sample test images")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2563EB", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#DC2626", linewidth=2)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Acc", color="#16A34A", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Acc", color="#9333EA", linewidth=2)
    ax2.set_title("Training & Validation Accuracy (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Training curves -> {save_path}")


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar=True, annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Confusion Matrix (Validation Set)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Ground Truth Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] Confusion matrix heatmap -> {save_path}")


def plot_sample_predictions(val_dataset, preds, probs, classes, save_path, num_samples=8):
    indices = np.random.choice(len(val_dataset), size=min(num_samples, len(val_dataset)), replace=False)
    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, idx in enumerate(indices):
        pil_img, true_class = val_dataset.get_raw_pil(idx)
        pred_idx = preds[idx]
        pred_class = classes[pred_idx]
        confidence = probs[idx][pred_idx] * 100
        is_correct = (pred_class == true_class)

        axes[i].imshow(pil_img)
        axes[i].axis("off")
        color = "#16A34A" if is_correct else "#DC2626"
        symbol = "✓" if is_correct else "✗"
        title = f"{symbol} Pred: {pred_class} ({confidence:.1f}%)\nTrue: {true_class}"
        axes[i].set_title(title, fontsize=11, fontweight="bold", color=color, pad=6)

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Test Predictions (MobileNetV2 Transfer Learning)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Sample predictions grid -> {save_path}")


def export_sample_images(val_dataset, output_dir, samples_per_class=2):
    os.makedirs(output_dir, exist_ok=True)
    saved_count = {cls_name: 0 for cls_name in val_dataset.classes}
    
    for idx in range(len(val_dataset)):
        pil_img, cls_name = val_dataset.get_raw_pil(idx)
        if saved_count[cls_name] < samples_per_class:
            img_filename = f"{cls_name}_sample_{saved_count[cls_name]+1}.png"
            pil_img.save(os.path.join(output_dir, img_filename))
            saved_count[cls_name] += 1
        if all(count >= samples_per_class for count in saved_count.values()):
            break
    print(f"[Saved] Exported demo test images -> {output_dir}/")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65)
    print("   DEEP LEARNING IMAGE CLASSIFIER - TRANSFER LEARNING PIPELINE")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f"Train samples per class: {args.samples_train} | Val samples per class: {args.samples_val}")
    print(f"Selected Classes: {DEFAULT_CLASSES}")

    # 1. Load Data
    t_start = time.time()
    train_loader, val_loader, classes = get_dataloaders(
        samples_per_class_train=args.samples_train,
        samples_per_class_val=args.samples_val,
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    print(f"Dataset ready! Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # 2. Initialize Model
    model = TransferClassifier(num_classes=len(classes), freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # 3. Training Loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()

        # Optional fine-tuning: unfreeze top convolutional layers
        if args.fine_tune_epoch > 0 and epoch == args.fine_tune_epoch:
            print("\n>>> Unfreezing top backbone convolutional blocks for fine-tuning...")
            model.unfreeze_top_layers(num_blocks=3)
            # Recreate optimizer with lower learning rate for backbone
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr}
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1, eta_min=1e-6)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}%")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "num_classes": len(classes),
                "img_size": args.img_size,
                "best_val_acc": best_val_acc,
                "history": history,
                "arch": "mobilenet_v2"
            }
            torch.save(checkpoint, best_model_path)

    total_training_time = time.time() - t_start
    print(f"\nTraining completed in {total_training_time:.1f}s. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # 4. Final Evaluation with Best Model
    print("\n--- Running Final Evaluation on Validation Set ---")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_loss, val_acc, val_preds, val_targets, val_probs = evaluate(model, val_loader, criterion, device)

    # Classification Report
    report = classification_report(val_targets, val_preds, target_names=classes, output_dict=True)
    report_text = classification_report(val_targets, val_preds, target_names=classes)
    print("\nClassification Report:\n" + report_text)

    # Save metrics JSON
    metrics_summary = {
        "best_val_accuracy": float(best_val_acc),
        "final_val_loss": float(val_loss),
        "total_training_time_seconds": float(total_training_time),
        "classes": classes,
        "classification_report": report
    }
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[Saved] Evaluation metrics -> {metrics_path}")

    # 5. Visualizations
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_curves(history, curves_path)

    cm = confusion_matrix(val_targets, val_preds)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path)

    samples_path = os.path.join(args.output_dir, "sample_predictions.png")
    plot_sample_predictions(val_loader.dataset, val_preds, val_probs, classes, samples_path)

    # 6. Export Demo Test Images
    export_sample_images(val_loader.dataset, args.samples_dir)

    print("\n" + "=" * 65)
    print("   ALL ARTIFACTS AND DELIVERABLES GENERATED SUCCESSFULLY!")
    print(f"   Model Checkpoint : {best_model_path}")
    print(f"   Visualizations   : {args.output_dir}/")
    print(f"   Sample Images    : {args.samples_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    main()

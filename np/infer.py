import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TORCH_HOME", "D:/torch_cache")

import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from model import TransferClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference using Saved Model")
    parser.add_argument("--image_path", type=str, default=None, help="Path to input image file (JPEG/PNG)")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--samples_dir", type=str, default="sample_images", help="Directory of sample images to run if --image_path is not given")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found at: {checkpoint_path}. Please run train.py first!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 96)

    model = TransferClassifier(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, img_size


def predict(image_path: str, model, classes, img_size: int, device: torch.device):
    # Standard normalization transform for inference
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pil_img = Image.open(image_path).convert("RGB")
    tensor_img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probabilities))
    top_class = classes[top_idx]
    top_confidence = probabilities[top_idx] * 100

    return top_class, top_confidence, probabilities


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("      DEEP LEARNING MODEL INFERENCE ENGINE")
    print("=" * 60)
    print(f"Loading checkpoint: {args.model_path}")
    model, classes, img_size = load_model(args.model_path, device)
    print(f"Model loaded successfully! Supported classes: {classes}\n")

    # Determine images to process
    if args.image_path:
        images_to_process = [args.image_path]
    else:
        if os.path.exists(args.samples_dir):
            images_to_process = [
                os.path.join(args.samples_dir, f)
                for f in os.listdir(args.samples_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        else:
            images_to_process = []

    if not images_to_process:
        print(f"No image specified and no images found in '{args.samples_dir}'.")
        print("Usage: python infer.py --image_path path/to/image.png")
        return

    print(f"Processing {len(images_to_process)} image(s):\n")
    for img_path in images_to_process:
        top_class, top_conf, probs = predict(img_path, model, classes, img_size, device)
        print(f"Image: {os.path.basename(img_path)}")
        print(f"  -> Predicted Class : {top_class.upper()}")
        print(f"  -> Confidence      : {top_conf:.2f}%")
        print("  -> Probability Distribution:")
        for cls_name, prob in zip(classes, probs):
            bar = "█" * int(prob * 25)
            print(f"     {cls_name:>10}: {prob*100:5.1f}%  {bar}")
        print("-" * 50)


if __name__ == "__main__":
    main()

# train_yolo11_cls.py
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

def ensure_val_split(root: Path, val_ratio: float = 0.2) -> Path:
    """
    Ensures a 'val' split exists. If not, creates data/_autoval by sampling
    val_ratio from data/train for each class.
    Returns the Path to the validation folder to use.
    """
    train_dir = root / "train"
    val_dir = root / "val"
    if val_dir.exists():
        return val_dir

    auto_val = root / "_autoval"
    if auto_val.exists():
        shutil.rmtree(auto_val)
    auto_val.mkdir(parents=True, exist_ok=True)

    # Create class subfolders
    classes = [p.name for p in (train_dir).iterdir() if p.is_dir()]
    for cls in classes:
        (auto_val / cls).mkdir(parents=True, exist_ok=True)

        imgs = [p for p in (train_dir / cls).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.is_file()]
        if not imgs:
            continue
        k = max(1, int(len(imgs) * val_ratio))
        sample = random.sample(imgs, k)
        for src in sample:
            dst = auto_val / cls / src.name
            shutil.copy2(src, dst)

    return auto_val

def main():
    random.seed(42)

    # Dataset root
    data_root = Path("data")
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise SystemExit("data/train not found. Create data/train/real and data/train/fake with images.")

    # Ensure we have a validation split (data/val or auto one)
    val_dir = ensure_val_split(data_root, val_ratio=0.2)

    # Load ImageNet-pretrained YOLO11 classification model
    model = YOLO("yolo11n-cls.pt")

    # Train
    results = model.train(
        data=str(train_dir),   # train path with class subfolders
        val=str(val_dir),      # explicit validation path
        imgsz=224,
        epochs=15,             # increase if you add more images (e.g., 20–30)
        batch=16,
        workers=0,             # Windows-friendly; raise on Linux/Mac if desired
        device="auto",
        project="runs_cls",
        name="y11n_real_fake"
    )

    # Find the latest best.pt and copy it to a friendly filename
    weights = list((Path("runs_cls")).rglob("weights/best.pt"))
    if not weights:
        raise SystemExit("Could not locate best.pt in runs_cls/*/weights/")
    # pick the most recently modified
    weights.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    best_path = weights[0]
    out_path = Path("yolo11n-cls-real-fake.pt")
    shutil.copy2(best_path, out_path)
    print(f"Model saved as {out_path} (copied from {best_path})")

if __name__ == "__main__":
    main()

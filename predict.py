# predict_yolo11_cls.py
import argparse, json
from ultralytics import YOLO
import cv2
from PIL import Image

def predict_on_image(model_path: str, image_path: str):
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=224, verbose=False)
    r = results[0]
    classes = r.names
    probs = r.probs.data.tolist()
    pred_idx = int(r.probs.top1)
    pred_label = classes[pred_idx] if isinstance(classes, dict) else classes[pred_idx]
    out = {
        "label": pred_label,
        "probabilities": { (classes[i] if isinstance(classes, dict) else classes[i]) : float(p)
                           for i, p in enumerate(probs) }
    }
    print(json.dumps(out, indent=2))

def predict_from_webcam(model_path: str, cam_index: int = 0):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"❌ Could not open webcam index {cam_index}")

    print("Press 'c' to capture & classify, 'q' to quit.")
    last_json = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Webcam (press 'c' to classify)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('c'):
            # BGR -> RGB, then PIL image for YOLO classification
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            results = model.predict(source=pil_img, imgsz=224, verbose=False)
            r = results[0]
            classes = r.names
            probs = r.probs.data.tolist()
            pred_idx = int(r.probs.top1)
            pred_label = classes[pred_idx] if isinstance(classes, dict) else classes[pred_idx]
            last_json = {
                "label": pred_label,
                "probabilities": { (classes[i] if isinstance(classes, dict) else classes[i]) : float(p)
                                   for i, p in enumerate(probs) }
            }
            print(json.dumps(last_json, indent=2))

            # Overlay result on the frame
            overlay = frame.copy()
            conf = max(last_json["probabilities"].values())
            txt = f"{pred_label} ({conf:.2f})"
            cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Webcam (press 'c' to classify)", overlay)

    cap.release()
    cv2.destroyAllWindows()
    if last_json is None:
        print("No capture performed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="Path to trained model (e.g., yolo11n-cls-real-fake.pt)")
    ap.add_argument("image", nargs="?", type=str, help="Path to image file (omit if using --webcam)")
    ap.add_argument("--webcam", action="store_true", help="Use webcam instead of an image")
    ap.add_argument("--cam-index", type=int, default=0, help="Webcam index (default 0)")
    args = ap.parse_args()

    if args.webcam:
        predict_from_webcam(args.model, cam_index=args.cam_index)
    else:
        if not args.image:
            raise SystemExit("Provide an image path or use --webcam.")
        predict_on_image(args.model, args.image)

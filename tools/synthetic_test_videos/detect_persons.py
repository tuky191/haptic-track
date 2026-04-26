#!/usr/bin/env python3
"""Run EfficientDet-Lite2 (same model as the app) on a frame and print
person bboxes in normalized coords. Used to pick a clean lockBox without a
GUI annotator."""
import sys
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

if len(sys.argv) != 2:
    print("usage: detect_persons.py <frame.png>")
    sys.exit(1)

frame_path = Path(sys.argv[1])
model_path = Path(__file__).parent.parent.parent / "app/src/main/assets/efficientdet-lite2-fp16.tflite"

base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=20,
)
detector = vision.ObjectDetector.create_from_options(options)
img = mp.Image.create_from_file(str(frame_path))
result = detector.detect(img)

w, h = img.width, img.height
print(f"image: {w}x{h}")
print(f"detections: {len(result.detections)}")
for i, det in enumerate(result.detections):
    bbox = det.bounding_box
    cat = det.categories[0]
    if cat.category_name != "person":
        continue
    nl = bbox.origin_x / w
    nt = bbox.origin_y / h
    nr = (bbox.origin_x + bbox.width) / w
    nb = (bbox.origin_y + bbox.height) / h
    area = (bbox.width / w) * (bbox.height / h)
    print(f"  #{i} person score={cat.score:.3f} box=[{nl:.3f},{nt:.3f},{nr:.3f},{nb:.3f}] area={area:.4f}")

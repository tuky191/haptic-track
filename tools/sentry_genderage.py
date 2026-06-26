#!/usr/bin/env python3
"""Phase 1 de-risk: validate InsightFace genderage on real HapticTrack faces.

Detects faces with YuNet (bbox + 5 landmarks), aligns to the ArcFace template,
runs genderage, prints gender + age. Then a SIZE-DEGRADATION sweep: takes the
largest face, downscales the source region to simulate distance/zoom, and reports
where the prediction stops being stable — that pixel size sets the zoom-to-inspect
threshold for the sentry.

Usage: .venv/bin/python sentry_genderage.py [image.png ...]   (defaults to sentry_faces/raw_*.png)
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent / "sentry_faces"
GENDERAGE = str(HERE / "genderage.onnx")
YUNET = str(HERE / "face_detection_yunet_2023mar.onnx")

# ArcFace 5-point template (112x112), scaled to 96 for genderage.
ARCFACE_112 = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
TEMPLATE_96 = ARCFACE_112 * (96.0 / 112.0)

_ga = ort.InferenceSession(GENDERAGE, providers=["CPUExecutionProvider"])


def detect_faces(bgr):
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(YUNET, "", (w, h), score_threshold=0.6)
    det.setInputSize((w, h))
    n, faces = det.detect(bgr)
    return faces if faces is not None else np.empty((0, 15))


def align96(bgr, landmarks5):
    M, _ = cv2.estimateAffinePartial2D(landmarks5.astype(np.float32), TEMPLATE_96, method=cv2.LMEDS)
    return cv2.warpAffine(bgr, M, (96, 96), borderValue=0)


def genderage(face96_bgr):
    blob = cv2.dnn.blobFromImage(face96_bgr, 1.0, (96, 96), (0, 0, 0), swapRB=True)
    pred = _ga.run(None, {"data": blob})[0][0]
    gender = "M" if pred[1] > pred[0] else "F"
    gconf = float(abs(pred[1] - pred[0]))
    age = int(round(float(pred[2]) * 100))
    return gender, age, gconf


def classify_image(path):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"  {path}: unreadable"); return None
    faces = detect_faces(bgr)
    out = []
    for f in faces:
        box = f[:4]; lmk = f[4:14].reshape(5, 2); det_score = f[14]
        face96 = align96(bgr, lmk)
        g, a, gc = genderage(face96)
        face_px = int(max(box[2], box[3]))
        out.append((g, a, gc, face_px, det_score))
    return bgr, faces, out


def main():
    imgs = sys.argv[1:] or sorted(glob.glob(str(HERE / "raw_*.png")))
    print(f"genderage de-risk on {len(imgs)} frames\n")
    biggest = None  # (img path, face row) for the size sweep
    for p in imgs:
        r = classify_image(p)
        if r is None: continue
        bgr, faces, out = r
        name = Path(p).name
        if not out:
            print(f"{name}: no face detected"); continue
        for (g, a, gc, fpx, ds) in out:
            print(f"{name}: gender={g} age={a}  (genderConf={gc:.2f}, facePx={fpx}, detScore={ds:.2f})")
            if biggest is None or fpx > biggest[2]:
                biggest = (p, faces, fpx)

    if biggest is None:
        print("\nno faces for size sweep"); return
    # SIZE-DEGRADATION SWEEP on the largest face's frame
    p, faces, fpx = biggest
    bgr = cv2.imread(p)
    print(f"\n=== size-degradation sweep (source {Path(p).name}, native face {fpx}px) ===")
    print("simulate distance: downscale whole frame, re-detect+classify the face")
    base_g, base_a = None, None
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.18, 0.12):
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        r = classify_image_arr(small)
        if not r:
            print(f"  scale={scale:.2f} facePx~{int(fpx*scale)}: NO FACE DETECTED"); continue
        g, a, gc, dpx, ds = max(r, key=lambda x: x[3])
        if base_g is None: base_g, base_a = g, a
        flag = "" if (g == base_g and abs(a - base_a) <= 8) else "  <-- DIVERGES"
        print(f"  scale={scale:.2f} facePx~{dpx}: gender={g} age={a} (conf={gc:.2f}){flag}")


def classify_image_arr(bgr):
    faces = detect_faces(bgr)
    out = []
    for f in faces:
        lmk = f[4:14].reshape(5, 2)
        g, a, gc = genderage(align96(bgr, lmk))
        out.append((g, a, gc, int(max(f[2], f[3])), f[14]))
    return out


if __name__ == "__main__":
    main()

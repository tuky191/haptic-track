#!/usr/bin/env python3
"""Research: online classifier for object identity discrimination.

Simulates approach A from issue #50: train a linear classifier on embeddings
collected during tracking, then use classifier confidence for reacquisition
instead of raw cosine similarity.

The hypothesis: even with weak MobileNetV3 embeddings, a learned decision
boundary (positive vs negative examples from the scene) should discriminate
better than raw cosine similarity to a gallery centroid.

Usage:
    cd tools && .venv/bin/python research_online_classifier.py
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Helpers ──

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def crop_image(img, box):
    w, h = img.size
    return img.crop((int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)))


def embed_crop(embedder, img, box):
    import mediapipe as mp
    crop = crop_image(img, box)
    if crop.width < 1 or crop.height < 1:
        return None
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop))
    result = embedder.embed(mp_image)
    return np.array(result.embeddings[0].embedding, dtype=np.float32)


# ── Online classifier approaches ──

class CentroidMatcher:
    """Baseline: centroid (L2-normalized mean) + cosine similarity."""

    def __init__(self):
        self.positives = []
        self.name = "Centroid"

    def add_positive(self, emb):
        self.positives.append(emb)

    def add_negative(self, emb):
        pass  # not used

    def score(self, emb):
        if not self.positives:
            return 0.0
        centroid = np.mean(self.positives, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        return cosine_sim(centroid, emb)


class BestOfGalleryMatcher:
    """Current approach: max cosine sim to any gallery embedding."""

    def __init__(self):
        self.positives = []
        self.name = "BestOfGallery"

    def add_positive(self, emb):
        self.positives.append(emb)

    def add_negative(self, emb):
        pass

    def score(self, emb):
        if not self.positives:
            return 0.0
        return max(cosine_sim(p, emb) for p in self.positives)


class LinearSVMMatcher:
    """Online linear SVM trained on positive/negative embeddings."""

    def __init__(self):
        self.positives = []
        self.negatives = []
        self._model = None
        self.name = "LinearSVM"

    def add_positive(self, emb):
        self.positives.append(emb)
        self._model = None

    def add_negative(self, emb):
        self.negatives.append(emb)
        self._model = None

    def _train(self):
        if len(self.positives) < 1 or len(self.negatives) < 1:
            return False
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler

        X = np.vstack(self.positives + self.negatives)
        y = np.array([1]*len(self.positives) + [0]*len(self.negatives))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LinearSVC(C=1.0, max_iter=1000)
        self._model.fit(X_scaled, y)
        return True

    def score(self, emb):
        if self._model is None:
            if not self._train():
                # Fall back to centroid
                if self.positives:
                    centroid = np.mean(self.positives, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    return cosine_sim(centroid, emb)
                return 0.0

        X = self._scaler.transform(emb.reshape(1, -1))
        # Use decision_function for continuous score
        return float(self._model.decision_function(X)[0])


class LogisticMatcher:
    """Online logistic regression — gives calibrated probabilities."""

    def __init__(self):
        self.positives = []
        self.negatives = []
        self._model = None
        self.name = "LogisticReg"

    def add_positive(self, emb):
        self.positives.append(emb)
        self._model = None

    def add_negative(self, emb):
        self.negatives.append(emb)
        self._model = None

    def _train(self):
        if len(self.positives) < 1 or len(self.negatives) < 1:
            return False
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.vstack(self.positives + self.negatives)
        y = np.array([1]*len(self.positives) + [0]*len(self.negatives))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(C=1.0, max_iter=1000)
        self._model.fit(X_scaled, y)
        return True

    def score(self, emb):
        if self._model is None:
            if not self._train():
                if self.positives:
                    centroid = np.mean(self.positives, axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    return cosine_sim(centroid, emb)
                return 0.0

        X = self._scaler.transform(emb.reshape(1, -1))
        return float(self._model.predict_proba(X)[0, 1])


class PrototypeMatcher:
    """Prototype network: score = sim(candidate, pos_centroid) - sim(candidate, neg_centroid)."""

    def __init__(self):
        self.positives = []
        self.negatives = []
        self.name = "Prototype"

    def add_positive(self, emb):
        self.positives.append(emb)

    def add_negative(self, emb):
        self.negatives.append(emb)

    def score(self, emb):
        if not self.positives:
            return 0.0

        pos_centroid = np.mean(self.positives, axis=0)
        pos_centroid = pos_centroid / (np.linalg.norm(pos_centroid) + 1e-8)
        pos_sim = cosine_sim(pos_centroid, emb)

        if not self.negatives:
            return pos_sim

        neg_centroid = np.mean(self.negatives, axis=0)
        neg_centroid = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-8)
        neg_sim = cosine_sim(neg_centroid, emb)

        # Return margin: how much closer to positive than negative
        return pos_sim - neg_sim


# ── Test scenarios ──

def test_apple_scenario(embedder):
    """Test with reacquisition_basic: single apple, different angles."""
    print("\n" + "=" * 70)
    print("SCENARIO: Single apple — same object from different angles")
    print("=" * 70)

    fixtures = Path("test_fixtures/reacquisition_basic")
    scenario = json.loads((fixtures / "scenario.json").read_text())
    frames_dir = fixtures / "frames"

    # Embed all annotated crops
    crops = {}
    for frame_ann in scenario["frames"]:
        frame_path = frames_dir / frame_ann["frame"]
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        for obj in frame_ann["objects"]:
            emb = embed_crop(embedder, img, obj["box"])
            if emb is not None:
                key = f"{obj['object_id']}@{frame_ann['frame']}"
                crops[key] = {"id": obj["object_id"], "emb": emb}

    print(f"  Computed {len(crops)} embeddings")

    # Simulate: lock on frame 1, accumulate through frames, test reacq
    lock_emb = crops["apple@frame_0001.png"]["emb"]
    frame2_emb = crops["apple@frame_0002.png"]["emb"]
    frame10_emb = crops["apple@frame_0010.png"]["emb"]
    frame30_emb = crops["apple@frame_0030.png"]["emb"]

    # For this scenario we don't have negatives from the same scene,
    # so generate synthetic negatives by adding noise
    rng = np.random.RandomState(42)
    synthetic_negatives = []
    for _ in range(5):
        neg = rng.randn(len(lock_emb)).astype(np.float32)
        neg = neg / (np.linalg.norm(neg) + 1e-8)
        synthetic_negatives.append(neg)

    matchers = [
        BestOfGalleryMatcher(),
        CentroidMatcher(),
        PrototypeMatcher(),
        LinearSVMMatcher(),
        LogisticMatcher(),
    ]

    # Phase 1: only lock embedding (what we have at lock time)
    for m in matchers:
        m.add_positive(lock_emb)
        for neg in synthetic_negatives:
            m.add_negative(neg)

    print("\n  Phase 1: lock embedding only (+ 5 synthetic negatives)")
    print(f"  {'Matcher':<16} {'frame2':>8} {'frame10':>8} {'frame30':>8}")
    for m in matchers:
        s2 = m.score(frame2_emb)
        s10 = m.score(frame10_emb)
        s30 = m.score(frame30_emb)
        print(f"  {m.name:<16} {s2:>8.3f} {s10:>8.3f} {s30:>8.3f}")

    # Phase 2: add frame2 as accumulated positive
    for m in matchers:
        m.add_positive(frame2_emb)

    print("\n  Phase 2: lock + frame2 positives")
    print(f"  {'Matcher':<16} {'frame10':>8} {'frame30':>8}")
    for m in matchers:
        s10 = m.score(frame10_emb)
        s30 = m.score(frame30_emb)
        print(f"  {m.name:<16} {s10:>8.3f} {s30:>8.3f}")


def test_two_cars_scenario(embedder):
    """Test with two_cars: dark car vs red car."""
    print("\n" + "=" * 70)
    print("SCENARIO: Two cars — lock dark car, reject red car")
    print("=" * 70)

    fixtures = Path("test_fixtures/two_cars")
    scenario = json.loads((fixtures / "scenario.json").read_text())
    frames_dir = fixtures / "frames"

    crops = {}
    for frame_ann in scenario["frames"]:
        frame_path = frames_dir / frame_ann["frame"]
        if not frame_path.exists():
            continue
        img = Image.open(frame_path).convert("RGB")
        for obj in frame_ann["objects"]:
            emb = embed_crop(embedder, img, obj["box"])
            if emb is not None:
                key = f"{obj['object_id']}@{frame_ann['frame']}"
                crops[key] = {"id": obj["object_id"], "emb": emb}

    print(f"  Computed {len(crops)} embeddings")

    dark_lock = crops["dark_car@190950_528_LOCK.png"]["emb"]
    dark_reacq = crops["dark_car@191004_442_REACQUIRE.png"]["emb"]
    red_car = crops["red_car@191011_074_REACQUIRE.png"]["emb"]

    matchers = [
        BestOfGalleryMatcher(),
        CentroidMatcher(),
        PrototypeMatcher(),
        LinearSVMMatcher(),
        LogisticMatcher(),
    ]

    # Phase 1: just lock embedding, no negatives
    for m in matchers:
        m.add_positive(dark_lock)

    print("\n  Phase 1: lock embedding only, no negatives")
    print(f"  {'Matcher':<16} {'dark_reacq':>12} {'red_car':>12} {'correct':>8}")
    for m in matchers:
        s_dark = m.score(dark_reacq)
        s_red = m.score(red_car)
        correct = "YES" if s_dark > s_red else "NO"
        print(f"  {m.name:<16} {s_dark:>12.3f} {s_red:>12.3f} {correct:>8}")

    # Phase 2: add red_car as negative (simulating scene context)
    matchers2 = [
        BestOfGalleryMatcher(),
        CentroidMatcher(),
        PrototypeMatcher(),
        LinearSVMMatcher(),
        LogisticMatcher(),
    ]
    for m in matchers2:
        m.add_positive(dark_lock)
        m.add_negative(red_car)
        # Add some synthetic negatives too
        rng = np.random.RandomState(42)
        for _ in range(3):
            neg = rng.randn(len(dark_lock)).astype(np.float32)
            neg = neg / (np.linalg.norm(neg) + 1e-8)
            m.add_negative(neg)

    print("\n  Phase 2: lock positive + red_car negative + 3 synthetic negatives")
    print(f"  {'Matcher':<16} {'dark_reacq':>12} {'red_car':>12} {'correct':>8}")
    for m in matchers2:
        s_dark = m.score(dark_reacq)
        s_red = m.score(red_car)
        correct = "YES" if s_dark > s_red else "NO"
        print(f"  {m.name:<16} {s_dark:>12.3f} {s_red:>12.3f} {correct:>8}")

    # Phase 3: simulate tracking — lock, see dark car again, then test
    matchers3 = [
        BestOfGalleryMatcher(),
        CentroidMatcher(),
        PrototypeMatcher(),
        LinearSVMMatcher(),
        LogisticMatcher(),
    ]
    for m in matchers3:
        m.add_positive(dark_lock)
        # Simulate: during VT, we see dark_reacq from slightly different angles
        # Generate augmented positives by adding small noise
        rng = np.random.RandomState(42)
        for _ in range(4):
            aug = dark_lock + rng.randn(len(dark_lock)).astype(np.float32) * 0.05
            aug = aug / (np.linalg.norm(aug) + 1e-8)
            m.add_positive(aug)
        m.add_negative(red_car)
        for _ in range(3):
            neg = rng.randn(len(dark_lock)).astype(np.float32)
            neg = neg / (np.linalg.norm(neg) + 1e-8)
            m.add_negative(neg)

    print("\n  Phase 3: 5 positive (lock + 4 augmented) + red_car neg + 3 synthetic neg")
    print(f"  {'Matcher':<16} {'dark_reacq':>12} {'red_car':>12} {'correct':>8}")
    for m in matchers3:
        s_dark = m.score(dark_reacq)
        s_red = m.score(red_car)
        correct = "YES" if s_dark > s_red else "NO"
        print(f"  {m.name:<16} {s_dark:>12.3f} {s_red:>12.3f} {correct:>8}")


def test_classifier_timing():
    """Measure how fast online training + inference is."""
    print("\n" + "=" * 70)
    print("CLASSIFIER TIMING (on-device feasibility)")
    print("=" * 70)

    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    dim = 1280  # MobileNetV3 dimension

    for n_pos, n_neg in [(5, 5), (10, 10), (12, 20), (12, 50)]:
        positives = rng.randn(n_pos, dim).astype(np.float32)
        negatives = rng.randn(n_neg, dim).astype(np.float32)
        X = np.vstack([positives, negatives])
        y = np.array([1]*n_pos + [0]*n_neg)
        test_sample = rng.randn(1, dim).astype(np.float32)

        # SVM
        t0 = time.perf_counter()
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        svm = LinearSVC(C=1.0, max_iter=1000)
        svm.fit(X_s, y)
        svm_train = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            scaler.transform(test_sample)
            svm.decision_function(scaler.transform(test_sample))
        svm_infer = (time.perf_counter() - t0) * 1000 / 100

        # Logistic
        t0 = time.perf_counter()
        scaler2 = StandardScaler()
        X_s2 = scaler2.fit_transform(X)
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_s2, y)
        lr_train = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(100):
            lr.predict_proba(scaler2.transform(test_sample))
        lr_infer = (time.perf_counter() - t0) * 1000 / 100

        print(f"  {n_pos}pos + {n_neg}neg:")
        print(f"    SVM:      train={svm_train:.1f}ms  infer={svm_infer:.3f}ms")
        print(f"    Logistic: train={lr_train:.1f}ms  infer={lr_infer:.3f}ms")


def main():
    import mediapipe as mp

    model_path = str(
        Path(__file__).parent.parent
        / "app/src/main/assets/mobilenet_v3_large_embedder.tflite"
    )
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=False
    )
    embedder = mp.tasks.vision.ImageEmbedder.create_from_options(options)

    test_apple_scenario(embedder)
    test_two_cars_scenario(embedder)
    test_classifier_timing()

    embedder.close()

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
  The key question: can a learned classifier on MobileNetV3 embeddings
  discriminate objects that raw cosine similarity cannot?

  If YES → implement online classifier on-device (approach A)
  If NO  → need better base features (DINOv2, approach C)
  If PARTIALLY → combine: better model + online classifier
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive tool to annotate objects in extracted frames for test scenarios.

Opens each frame, lets you draw bounding boxes and assign object IDs.
Outputs a scenario JSON file that the benchmark and Android tests consume.

Usage:
    python annotate_scenario.py <frames_dir> <output_json>

Controls:
    - Click and drag to draw a bounding box
    - Type an object ID when prompted (e.g. "red_cup", "white_cup")
    - Press 'n' to move to next frame
    - Press 's' to skip frame (no annotations)
    - Press 'q' to finish and save
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


class Annotator:
    def __init__(self, frames_dir: str):
        self.frames = sorted(Path(frames_dir).glob("frame_*.png"))
        if not self.frames:
            print(f"No frames found in {frames_dir}", file=sys.stderr)
            sys.exit(1)
        self.annotations = []
        self.drawing = False
        self.start_pt = None
        self.end_pt = None
        self.current_boxes = []
        self.img = None
        self.display = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.end_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_pt = (x, y)
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_pt = (x, y)
            if self.start_pt and self.end_pt:
                self._prompt_label()

    def _redraw(self):
        self.display = self.img.copy()
        # Draw existing boxes
        for box in self.current_boxes:
            x1, y1, x2, y2 = box["box_px"]
            cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display, box["object_id"], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Draw current selection
        if self.start_pt and self.end_pt:
            cv2.rectangle(self.display, self.start_pt, self.end_pt, (0, 255, 255), 2)
        cv2.imshow("Annotate", self.display)

    def _prompt_label(self):
        x1, y1 = self.start_pt
        x2, y2 = self.end_pt
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return
        # Normalize coordinates
        h, w = self.img.shape[:2]
        box_norm = [
            min(x1, x2) / w, min(y1, y2) / h,
            max(x1, x2) / w, max(y1, y2) / h
        ]
        box_px = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        object_id = input(f"  Object ID for box [{box_norm[0]:.2f},{box_norm[1]:.2f},{box_norm[2]:.2f},{box_norm[3]:.2f}]: ").strip()
        if object_id:
            self.current_boxes.append({
                "object_id": object_id,
                "box": box_norm,
                "box_px": box_px,
            })
            self._redraw()

    def run(self):
        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate", self.mouse_callback)

        print(f"Annotating {len(self.frames)} frames")
        print("Draw boxes, type IDs in terminal. 'n'=next frame, 's'=skip, 'q'=done\n")

        for i, frame_path in enumerate(self.frames):
            self.img = cv2.imread(str(frame_path))
            if self.img is None:
                continue
            self.current_boxes = []
            self.display = self.img.copy()

            print(f"Frame {i+1}/{len(self.frames)}: {frame_path.name}")
            cv2.imshow("Annotate", self.display)

            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('n'):
                    break
                elif key == ord('s'):
                    self.current_boxes = []
                    break
                elif key == ord('q'):
                    self._save_frame(frame_path)
                    cv2.destroyAllWindows()
                    return self.annotations

            self._save_frame(frame_path)

        cv2.destroyAllWindows()
        return self.annotations

    def _save_frame(self, frame_path):
        if self.current_boxes:
            self.annotations.append({
                "frame": frame_path.name,
                "objects": [
                    {"object_id": b["object_id"], "box": b["box"]}
                    for b in self.current_boxes
                ]
            })


def save_scenario(annotations: list, output_path: str):
    # Collect all unique object IDs
    all_ids = set()
    for frame in annotations:
        for obj in frame["objects"]:
            all_ids.add(obj["object_id"])

    scenario = {
        "description": "TODO: describe this test scenario",
        "object_ids": sorted(all_ids),
        "frames": annotations,
        "expected_reacquisitions": [
            {
                "description": "TODO: describe what should happen",
                "lock_on": {"object_id": "TODO", "frame": "TODO"},
                "should_reacquire": {"object_id": "TODO", "frame": "TODO"},
                "should_not_reacquire": []
            }
        ]
    }

    Path(output_path).write_text(json.dumps(scenario, indent=2))
    print(f"\nSaved scenario with {len(annotations)} annotated frames → {output_path}")
    print("Edit the 'expected_reacquisitions' section to define test cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate objects in frames")
    parser.add_argument("frames_dir", help="Directory with extracted frames")
    parser.add_argument("output_json", help="Output scenario JSON path")
    args = parser.parse_args()

    annotator = Annotator(args.frames_dir)
    annotations = annotator.run()
    save_scenario(annotations, args.output_json)

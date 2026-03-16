"""
SSS-TT Inference Script — real-time and batch prediction.

Usage (single video):
    python scripts/inference.py \
        --video_path /path/to/infant_clip.mp4 \
        --checkpoint checkpoints/sss-tt/best.pth \
        --output_path results/prediction.json

Usage (camera stream):
    python scripts/inference.py \
        --stream rtsp://nicu-camera:554/stream \
        --checkpoint checkpoints/sss-tt/best.pth \
        --live

Usage (batch directory):
    python scripts/inference.py \
        --video_dir /data/new_clips \
        --checkpoint checkpoints/sss-tt/best.pth \
        --output_path results/batch_predictions.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.sss_tt import build_sss_tt
from src.data.preprocessing import RetinaFacePreprocessor
from src.models.coral_head import cumprobs_to_class_probs


PAIN_LABELS = {0: 'No Pain', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
PAIN_COLORS_BGR = {
    0: (0, 200, 0),       # green
    1: (0, 165, 255),     # orange
    2: (0, 100, 255),     # deep orange
    3: (0, 0, 220),       # red
}


def parse_args():
    p = argparse.ArgumentParser(description='SSS-TT Inference')
    p.add_argument('--video_path', type=str, default=None)
    p.add_argument('--video_dir', type=str, default=None)
    p.add_argument('--stream', type=str, default=None)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output_path', type=str, default='prediction.json')
    p.add_argument('--T', type=int, default=32)
    p.add_argument('--mc_passes', type=int, default=10)
    p.add_argument('--live', action='store_true')
    p.add_argument('--threshold_alert', type=int, default=2,
                   help='Pain level that triggers alert (default 2)')
    p.add_argument('--device', type=str, default='auto')
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    cfg = ckpt.get('config', {})
    model = build_sss_tt(cfg).to(device)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class SSTTPainMonitor:
    """
    Real-time pain monitoring wrapper for SSS-TT.

    Maintains a rolling buffer of T frames and runs inference
    at each buffer fill (non-blocking sliding window).
    """

    def __init__(self, model, preprocessor, T: int = 32,
                 mc_passes: int = 10, device=None):
        self.model = model
        self.preprocessor = preprocessor
        self.T = T
        self.mc_passes = mc_passes
        self.device = device or torch.device('cpu')
        self._frame_buffer = []
        self._last_result = None

    def push_frame(self, frame_bgr: np.ndarray) -> dict | None:
        """
        Push a new BGR frame. Returns prediction dict when buffer is full,
        else None. Implements a sliding window (50% overlap).
        """
        tensor = self.preprocessor.frame_to_tensor(
            self.preprocessor.detect_and_align(frame_bgr)
            or np.zeros((224, 224, 3), dtype=np.uint8)
        )
        self._frame_buffer.append(tensor)

        if len(self._frame_buffer) >= self.T:
            video = torch.stack(self._frame_buffer[-self.T:], dim=0)
            result = self._infer(video)
            # Slide by T//2 frames
            self._frame_buffer = self._frame_buffer[-(self.T // 2):]
            self._last_result = result
            return result
        return self._last_result

    @torch.no_grad()
    def _infer(self, video: torch.Tensor) -> dict:
        t0 = time.time()
        video = video.unsqueeze(0).to(self.device)             # (1, T, 3, 224, 224)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        result = raw_model.predict_with_confidence(
            video, mc_passes=self.mc_passes
        )
        latency = time.time() - t0
        pred = result['pred'].item()
        conf = result['confidence'].item()
        probs = result['class_probs'].squeeze(0).cpu().numpy().tolist()
        return {
            'pain_level': pred,
            'pain_label': PAIN_LABELS[pred],
            'confidence': round(conf, 3),
            'class_probs': [round(p, 3) for p in probs],
            'alert_level': result['alert_level'].item(),
            'latency_s': round(latency, 3),
        }

    def overlay_result(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw pain prediction overlay on frame for live display."""
        if result is None:
            return frame
        pain_level = result['pain_level']
        color = PAIN_COLORS_BGR[pain_level]
        h, w = frame.shape[:2]

        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Pain label
        cv2.putText(frame,
                    f"Pain: {result['pain_label']} (Level {pain_level})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Confidence bar
        conf = result['confidence']
        bar_w = int(conf * 250)
        cv2.rectangle(frame, (10, 42), (10 + bar_w, 58), (200, 200, 0), -1)
        cv2.rectangle(frame, (10, 42), (260, 58), (100, 100, 100), 1)
        cv2.putText(frame, f"Conf: {conf:.0%}",
                    (270, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1)

        # Alert indicator
        alert = result['alert_level']
        if alert == 2:
            cv2.putText(frame, "!! ALERT !!",
                        (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 0, 255), 2)
        elif alert == 1:
            cv2.putText(frame, "REVIEW",
                        (w - 100, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 165, 255), 2)

        return frame


def run_single(args, model, preprocessor, device) -> dict:
    """Process a single video file."""
    monitor = SSTTPainMonitor(model, preprocessor,
                              T=args.T, mc_passes=args.mc_passes,
                              device=device)
    video_tensor = preprocessor.process_video(args.video_path, T=args.T)
    result = monitor._infer(video_tensor)
    result['video_path'] = args.video_path
    return result


def run_batch(args, model, preprocessor, device) -> list:
    """Process all videos in a directory."""
    video_dir = Path(args.video_dir)
    videos = sorted(list(video_dir.rglob('*.mp4')) +
                    list(video_dir.rglob('*.avi')))
    print(f"Found {len(videos)} videos")

    results = []
    monitor = SSTTPainMonitor(model, preprocessor,
                              T=args.T, mc_passes=args.mc_passes,
                              device=device)
    for i, vpath in enumerate(videos):
        try:
            video_tensor = preprocessor.process_video(str(vpath), T=args.T)
            result = monitor._infer(video_tensor)
            result['video_path'] = str(vpath)
            results.append(result)
            print(f"[{i+1}/{len(videos)}] {vpath.name}: "
                  f"Pain={result['pain_label']} "
                  f"Conf={result['confidence']:.0%} "
                  f"({result['latency_s']}s)")
        except Exception as e:
            print(f"  ERROR processing {vpath.name}: {e}")
    return results


def run_live(args, model, preprocessor, device):
    """Live inference from camera stream or webcam."""
    src = args.stream or 0
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {src}")
        return

    monitor = SSTTPainMonitor(model, preprocessor,
                              T=args.T, mc_passes=args.mc_passes,
                              device=device)
    print("Live inference started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = monitor.push_frame(frame)
        display = monitor.overlay_result(frame.copy(), result)

        # FPS counter
        cv2.putText(display, f"{1/max(result['latency_s'], 0.001):.1f} FPS",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('SSS-TT Pain Monitor', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    preprocessor = RetinaFacePreprocessor(target_size=224)

    # Run inference
    if args.live or args.stream:
        run_live(args, model, preprocessor, device)

    elif args.video_dir:
        results = run_batch(args, model, preprocessor, device)
        os.makedirs(Path(args.output_path).parent, exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        pain_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for r in results:
            pain_counts[r['pain_level']] += 1
        print("\nBatch Summary:")
        for lvl, cnt in pain_counts.items():
            pct = cnt / len(results) * 100 if results else 0
            print(f"  {PAIN_LABELS[lvl]}: {cnt} ({pct:.1f}%)")
        print(f"Results saved to: {args.output_path}")

    elif args.video_path:
        result = run_single(args, model, preprocessor, device)
        print(json.dumps(result, indent=2))
        os.makedirs(Path(args.output_path).parent, exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {args.output_path}")

    else:
        print("ERROR: Provide --video_path, --video_dir, or --stream")


if __name__ == '__main__':
    main()

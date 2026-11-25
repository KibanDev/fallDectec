"""Extract identical YOLO pose observations for both versions of the app."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_app.pose_detector import MODEL_PATH

SAMPLE_FPS = 10


def frames(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        raise ValueError('Invalid video FPS')
    index = 0
    sample = 0
    while capture.grab():
        if index >= round(sample * fps / SAMPLE_FPS):
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f'Could not decode frame {index}')
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                frame = cv2.resize(frame, (round(w * 640 / max(h, w)),
                                           round(h * 640 / max(h, w))), interpolation=cv2.INTER_AREA)
            yield index / fps, frame
            sample += 1
        index += 1


def cache_video(video, model, device):
    capture = cv2.VideoCapture(str(video))
    rows, batch, times = [], [], []
    try:
        for timestamp, frame in frames(capture):
            times.append(timestamp)
            batch.append(frame)
            if len(batch) == 16:
                rows.extend(predict(model, device, times, batch))
                times.clear()
                batch.clear()
        if batch:
            rows.extend(predict(model, device, times, batch))
    finally:
        capture.release()
    if not rows:
        raise ValueError(f'Empty video: {video}')
    return rows


def predict(model, device, times, batch):
    results = model.predict(batch, imgsz=480, conf=.5, device=device, verbose=False,
                            rect=False)
    for timestamp, result in zip(times, results, strict=True):
        yield [timestamp, result.boxes.data.cpu().tolist(), result.keypoints.data.cpu().tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = YOLO(str(MODEL_PATH))
    videos = sorted(args.dataset.glob('Subject */*/*.mp4'))
    if not videos:
        raise ValueError(f'No GMDCSA24 recordings in {args.dataset}')
    for i, video in enumerate(videos, 1):
        cache = args.dataset / 'poses' / video.relative_to(args.dataset).with_suffix('.json')
        if not cache.exists():
            rows = cache_video(video, model, args.device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            partial = cache.with_suffix('.part')
            partial.write_text(json.dumps(rows, separators=(',', ':')), encoding='utf-8')
            partial.replace(cache)
        if i % 10 == 0 or i == len(videos):
            print(f'Pose cache {i}/{len(videos)}', flush=True)


if __name__ == '__main__':
    main()

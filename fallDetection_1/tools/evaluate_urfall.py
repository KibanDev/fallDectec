"""Compare both versions on URFD RGB clips using sequence-level labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from cache_pose import MODEL_PATH, ROOT, SAMPLE_FPS, cache_video
from train_pose import Clip, LearnedPose, baseline_engine, compare
from python_app.learned_pose import MODEL_PATH as LEARNED_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data'/'urfall')
    parser.add_argument('--baseline', type=Path, default=ROOT.parent/'fallDetection_0')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', type=Path, default=LEARNED_MODEL_PATH)
    parser.add_argument('--report', type=Path, default=ROOT/'results'/'urfall_comparison.json')
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    detector = YOLO(str(MODEL_PATH))
    model = LearnedPose.load(args.model)
    if model is None:
        raise FileNotFoundError('The active learned pose weights are required')
    with (args.data/'manifest.csv').open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    counts = Counter(row['sequence_label'] for row in rows)
    if set(counts) != {'fall', 'non_fall_adl'}:
        raise ValueError('The URFD manifest must contain both fall and ADL clips')
    with TemporaryDirectory(prefix='urfall-poses-') as temporary:
        dataset = []
        for index, row in enumerate(rows):
            video = args.data/row['video']
            if hashlib.sha256(video.read_bytes()).hexdigest() != row['rgb_sha256']:
                raise ValueError(f'RGB clip differs from manifest: {video}')
            cache = Path(temporary)/f'{index}.json'
            cache.write_text(json.dumps(cache_video(video, detector, args.device)), encoding='utf-8')
            dataset.append(Clip(video, cache, 0, row['sequence_label'] == 'fall', None, ()))
            print(f'URFD pose {index+1}/{len(rows)}: {video.name}', flush=True)
        summary, recordings = compare(dataset, model, baseline_engine(args.baseline), sequence_level=True)
    metrics = {name: {key: value for key, value in counts.items()
                      if key in ('detected_falls', 'adl_clips_alerted')}
               for name, counts in summary.items() if not name.endswith('_posture')}
    report = dict(dataset='UR Fall Detection RGB previews', sample_fps=SAMPLE_FPS,
                  policy=model.policy,
                  clip_counts=dict(counts), ground_truth='whole clip only; no onset timestamps',
                  sensor='none', results=metrics, recordings=recordings)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

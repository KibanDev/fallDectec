"""Train on GMDCSA24 subjects 1/2, calibrate on 3, and compare on 4."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
import types

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from cache_pose import ROOT, frames
from upfall_examples import evaluate_impact, examples as upfall_examples
from python_app.learned_pose import (FEATURE_VERSION, MODEL_PATH, POSTURES,
                                     LearnedPose, LearnedTrack, LinearModel, PoseHistory)
from python_app.main import build_parser
from python_app.posture_analyzer import analyze_posture
from python_app.scene_analyzer import SceneAnalyzer

# One source annotation omits the word "to" between its two timestamps.
INTERVAL = re.compile(r"([a-z]+(?:\s*\([a-z]+\))?)\s*\[\s*(\d+(?:\.\d+)?)(?:\s*to\s*|\s+)(\d+(?:\.\d+)?)\s*\]", re.I)
POSTURE_LABELS = {'standing': 0, 'walking': 0, 'sitting': 1, 'sleeping': 2}
CAUCA_SHA256 = 'ec766ee3c8108bacc67ba0fa140f981c0738d71546bfadc54c70ca8209c50ae6'
COCO = ('Nose', 'Left_Eye', 'Right_Eye', 'Left_Ear', 'Right_Ear', 'Left_Shoulder',
        'Right_Shoulder', 'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
        'Left_Hip', 'Right_Hip', 'Left_Knee', 'Right_Knee', 'Left_Ankle', 'Right_Ankle')


@dataclass(frozen=True)
class Clip:
    path: Path
    cache: Path
    subject: int
    is_fall: bool
    onset: float | None
    intervals: tuple


def clips(root):
    for subject in range(1, 5):
        directory = root / f'Subject {subject}'
        for kind in ('ADL', 'Fall'):
            lines = (directory / f'{kind}.csv').read_text(encoding='utf-8-sig').splitlines()[1:]
            for line in filter(str.strip, lines):
                filename = line.split(',', 1)[0].strip(' "')
                intervals = tuple((name.lower(), float(start), float(end))
                                  for name, start, end in INTERVAL.findall(line))
                falls = [start for label, start, _ in intervals
                         if label.startswith('fall') or label == 'bw']
                if not intervals or (falls and kind != 'Fall'):
                    raise ValueError(f'Invalid labels for {directory.name}/{kind}/{filename}: {line}')
                video = directory / kind / filename
                cache = root / 'poses' / video.relative_to(root).with_suffix('.json')
                yield Clip(video, cache, subject, kind == 'Fall', min(falls) if falls else None, intervals)


def posture_label(clip, timestamp):
    labels = {POSTURE_LABELS[label] for label, start, end in clip.intervals
              if label in POSTURE_LABELS and start+.12 <= timestamp <= end-.12}
    if len(labels) == 1:
        return labels.pop()
    return None


def pose_rows(clip):
    history = PoseHistory()
    for timestamp, boxes, points in json.loads(clip.cache.read_text(encoding='utf-8')):
        # The source labels identify the clip's participant, not other detections.
        if len(boxes) != 1 or len(points) != 1:
            history = PoseHistory()
            yield timestamp, None, None, 'UNKNOWN'
            continue
        static, motion = history.update(boxes[0], points[0], timestamp)
        yield timestamp, static, motion, analyze_posture(points[0]).status


def examples(dataset):
    posture, motion = [], []
    for clip in dataset:
        for timestamp, static, temporal, _ in pose_rows(clip):
            label = posture_label(clip, timestamp)
            if static is not None and label is not None:
                posture.append((static, label))
            if temporal is None:
                continue
            if not clip.is_fall or (clip.onset is not None and timestamp < clip.onset-.35):
                motion.append((temporal, 0))
            elif clip.onset is not None and .02 <= timestamp-clip.onset <= .85:
                motion.append((temporal, 1))
    return posture, motion


def cauca_examples(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != CAUCA_SHA256:
        raise ValueError('Unexpected CAUCAFall keypoint export checksum')
    sequences = defaultdict(list)
    with path.open(newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            if int(row['subject'].split('.')[-1]) <= 7:
                sequences[row['subject'], row['action']].append(row)
    posture, motion = [], []
    for (_, action), rows in sequences.items():
        history = PoseHistory()
        last_time = -1.
        for row in sorted(rows, key=lambda entry: int(entry['frame_idx'])):
            now = (int(row['frame_idx'])-1)/23
            if now-last_time < .095 or not float(row['detection_score']) >= .5:
                continue
            width, height = int(row['img_width']), int(row['img_height'])
            points = []
            for name in COCO:
                x, y = float(row[f'{name}_x']), float(row[f'{name}_y'])
                points.append([x, y, 1.] if 0 < x < width and 0 < y < height else [0., 0., 0.])
            visible = [point for point in points if point[2]]
            if len(visible) < 8:
                continue
            x, y, _ = np.asarray(visible).T
            margin = max(np.ptp(x), np.ptp(y)) * .08
            box = [max(0, x.min()-margin), max(0, y.min()-margin),
                   min(width, x.max()+margin), min(height, y.max()+margin)]
            static, temporal = history.update(box, points, now)
            last_time = now
            # A positive CAUCAFall frame is already on the ground. It cannot
            # label the fall onset. ADLs supply hard negatives for that head.
            if static is not None and row['is_fall'] == '1':
                posture.append((static, 2))
            if temporal is not None and not action.startswith('Fall'):
                motion.append((temporal, 0))
    return posture, motion


def fit(rows, classes, extra=(), hard_negatives=None):
    x, y = zip(*rows, strict=True)
    raw = np.asarray(x)
    labels = np.asarray(y, dtype=int)
    counts = np.bincount(labels, minlength=classes)
    if not counts.all():
        raise ValueError(f'Missing class examples: {counts}')
    mean, scale = raw.mean(axis=0), np.maximum(raw.std(axis=0), .05)
    weight = 1/(classes*counts[labels])
    if extra:
        additional, additional_labels = zip(*extra, strict=True)
        additional_labels = np.asarray(additional_labels, dtype=int)
        auxiliary_counts = np.bincount(additional_labels, minlength=classes)
        weight *= np.where(auxiliary_counts[labels] > 0, .8, 1.)
        weight = np.concatenate((weight, .2/(classes*auxiliary_counts[additional_labels])))
        raw = np.concatenate((raw, np.asarray(additional)))
        labels = np.concatenate((labels, additional_labels))
    if hard_negatives is not None:
        negative = labels == 0
        mass = weight[negative].sum()
        weight[negative] *= .1 + hard_negatives.probabilities(raw[negative])[:, 1] ** 2
        weight[negative] *= mass/weight[negative].sum()
    x = np.column_stack((np.clip((raw-mean)/scale, -6, 6), np.ones(len(raw))))
    targets = np.eye(classes)[labels]
    shape = (x.shape[1], classes)

    def loss(flat):
        coefficients = flat.reshape(shape)
        logits = x @ coefficients
        log_probs = logits-logsumexp(logits, axis=1, keepdims=True)
        penalty = .015 * coefficients[:-1]
        gradient = x.T @ ((np.exp(log_probs)-targets)*weight[:, None])
        gradient[:-1] += penalty
        return -(weight*log_probs[np.arange(len(labels)), labels]).sum() + (penalty*coefficients[:-1]).sum()/2, gradient.ravel()

    solution = minimize(loss, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B')
    if not solution.success:
        raise RuntimeError(solution.message)
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=solution.x.reshape(shape).tolist())


def calibrate(validation, model, motion_candidates):
    poses = [(static, target, observed) for clip in validation
             for timestamp, static, _, observed in pose_rows(clip)
             if static is not None and (target := posture_label(clip, timestamp)) is not None]
    posture_threshold = 1.1
    top_score = -1.
    for threshold in (1.1, .95, .85, .75, .65, .55):
        model.policy['posture_threshold'] = threshold
        rows = [(target, LearnedTrack().posture(model, static, observed))
                for static, target, observed in poses]
        score = statistics.mean(sum(outcome == name for target, outcome in rows if target == index)
                                / max(1, sum(target == index for target, _ in rows))
                                for index, name in enumerate(POSTURES))
        if score > top_score:
            top_score, posture_threshold = score, threshold

    observations = [(clip, [(t, motion) for t, _, motion, _ in pose_rows(clip) if motion is not None])
                    for clip in validation]
    results = []
    for index, candidate in enumerate(motion_candidates):
        motion = LinearModel(candidate)
        scored = [(clip, [(t, motion.probabilities(features)[1]) for t, features in rows])
                  for clip, rows in observations]
        for step in range(14, 39):
            threshold = step/40
            tp = fp = 0
            for clip, scores in scored:
                events = [t for t, score in scores if score >= threshold]
                if not clip.is_fall:
                    fp += bool(events)
                elif clip.onset is not None:
                    tp += any(clip.onset-.25 <= t <= clip.onset+1.5 for t in events)
            results.append((tp, fp, index, threshold))
    # Prefer fewer false alarms while retaining at least 90% of the best
    # observed validation recall across the fitted candidates.
    minimum = .9*max(row[0] for row in results)
    tp, fp, index, threshold = min((row for row in results if row[0] >= minimum),
                                   key=lambda row: (row[1], -row[0], row[2], -row[3]))
    model.motion = LinearModel(motion_candidates[index])
    policy = dict(posture_threshold=posture_threshold, motion_threshold=threshold)
    return policy, motion_candidates[index], dict(detected_falls=tp, adl_clips_alerted=fp,
                                                  hard_negative_passes=index)


def baseline_engine(root):
    package = types.ModuleType('baseline_pose_app')
    package.__path__ = [str(root.resolve() / 'python_app')]
    sys.modules[package.__name__] = package
    return importlib.import_module('baseline_pose_app.scene_analyzer').SceneAnalyzer


def compare(test, model, baseline, *, sequence_level=False):
    args = build_parser().parse_args([])
    audit = []
    postures = [Counter(), Counter()]
    for clip in test:
        scenes = [baseline(args), SceneAnalyzer(args, use_learned=False)]
        scenes[1].learned = model
        events = [[], []]
        capture = cv2.VideoCapture(str(clip.path))
        cached = json.loads(clip.cache.read_text(encoding='utf-8'))
        try:
            for (timestamp, frame), (cached_time, boxes, keypoints) in zip(frames(capture), cached, strict=True):
                if abs(timestamp-cached_time) > 1e-6:
                    raise ValueError(f'Cache timestamp mismatch: {clip.path}')
                for i, scene in enumerate(scenes):
                    posture, final, _, early, _, _ = scene.update(frame, boxes, keypoints, timestamp)
                    target = posture_label(clip, timestamp)
                    if target is not None:
                        postures[i][POSTURES[target], posture.status] += 1
                    if early:
                        events[i].append(['EARLY_WARNING', timestamp])
                    if final.warning:
                        events[i].append(['LYING_CHECK', timestamp])
        finally:
            capture.release()
        audit.append(dict(clip='/'.join(clip.path.parts[-3:]), is_fall=clip.is_fall, onset=clip.onset,
                          baseline=events[0], learned=events[1]))

    summary = {}
    for kind, counts in zip(('baseline', 'learned'), postures, strict=True):
        for category, matching in [('early', {'EARLY_WARNING'}), ('any', {'EARLY_WARNING', 'LYING_CHECK'})]:
            tp, fp, pre, delays = 0, 0, 0, []
            for row in audit:
                times = [t for label, t in row[kind] if label in matching]
                onset = row['onset']
                if not row['is_fall']:
                    fp += bool(times)
                elif sequence_level:
                    tp += bool(times)
                elif onset is not None:
                    pre += any(t < onset-.25 for t in times)
                    timed = [t-onset for t in times if onset-.25 <= t <= onset+1.5]
                    if timed:
                        tp += 1
                        delays.append(timed[0])
            summary[f'{kind}_{category}'] = dict(detected_falls=tp, adl_clips_alerted=fp,
                fall_clips_alerted_too_early=pre, median_delay_s=statistics.median(delays) if delays else None)
        summary[f'{kind}_posture'] = {name: {observed: count for (target, observed), count in counts.items()
                                            if name == target} for name in POSTURES}
    return summary, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data'/'gmdcsa')
    parser.add_argument('--baseline', type=Path, default=ROOT.parent/'fallDetection_0')
    parser.add_argument('--cauca', type=Path, default=ROOT/'data'/'caucafall'/'keypoints_raw.csv')
    parser.add_argument('--upfall', type=Path, help='Optional folder containing UP-Fall SUBJECT1–5 ZIP files')
    parser.add_argument('--reference-model', type=Path, default=MODEL_PATH,
                        help='Existing weights used for the UP-Fall held-out comparison')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.output = args.output or (ROOT/'results'/'upfall_candidate.json' if args.upfall else MODEL_PATH)
    args.report = args.report or ROOT/'results'/('upfall_comparison.json' if args.upfall else 'pose_comparison.json')
    cv2.setNumThreads(2)
    dataset = list(clips(args.data))
    train = [clip for clip in dataset if clip.subject in (1, 2)]
    validation = [clip for clip in dataset if clip.subject == 3]
    test = [clip for clip in dataset if clip.subject == 4]
    if len(dataset) != 160 or any(not clip.cache.exists() for clip in dataset):
        raise ValueError('Need all 160 annotated recordings and pose caches')
    posture, motion = examples(train)
    auxiliary_posture, auxiliary_motion = cauca_examples(args.cauca)
    upfall, upfall_sequences = upfall_examples(args.upfall) if args.upfall else ([], 0)
    training_motion = auxiliary_motion + upfall
    candidates = [fit(motion, 2, training_motion)]
    for _ in range(2):
        candidates.append(fit(motion, 2, training_motion, hard_negatives=LinearModel(candidates[-1])))
    data = dict(feature_version=FEATURE_VERSION, posture=fit(posture, len(POSTURES), auxiliary_posture),
                motion=candidates[0], policy={})
    model = LearnedPose(data)
    policy, data['motion'], validation_summary = calibrate(validation, model, candidates)
    model.policy.update(policy)
    print(f'Frozen policy: {model.policy}', flush=True)
    summary, recordings = compare(test, model, baseline_engine(args.baseline))
    report = dict(source=['GMDCSA24 v2.1', 'CAUCAFall keypoint export (subjects 1–7, training only)'],
                  split={'train': len(train), 'validation': len(validation), 'test': len(test)},
                  training_examples={'posture': dict(Counter(POSTURES[i] for _, i in posture)),
                                     'motion': dict(Counter(('non_fall', 'fall')[i] for _, i in motion))},
                  auxiliary_training_examples={'lying': len(auxiliary_posture), 'non_fall': len(auxiliary_motion)},
                  test_fall_clips=sum(clip.is_fall for clip in test),
                  test_fall_clips_with_onset=sum(clip.onset is not None for clip in test),
                  test_adl_clips=sum(not clip.is_fall for clip in test), policy=model.policy,
                  validation_summary=validation_summary,
                  results=summary, recordings=recordings)
    if args.upfall:
        report['source'].append('UP-Fall skeleton export, DOI 10.5281/zenodo.12773013, subjects 1–3')
        report['upfall_training'] = dict(sequences=upfall_sequences, impact_windows=len(upfall))
        reference = LearnedPose.load(args.reference_model)
        report['upfall_impact_check'] = dict(candidate=evaluate_impact(args.upfall, model),
                                           reference=evaluate_impact(args.upfall, reference) if reference else None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    args.output.write_text(json.dumps(data, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

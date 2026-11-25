"""Map the published UP-Fall BlazePose exports to camera motion features."""
import csv
import hashlib
from io import TextIOWrapper
import math
from zipfile import ZipFile

from python_app.learned_pose import LearnedTrack, PoseHistory

# BlazePose's 0-based anatomical indices, in COCO joint order.
JOINTS = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
CHECKSUMS = ('6c34ff6e898b42ecead633836206a5b0', '4db506e5a541ad2de84bf5009d440f13',
             '5f3d6b4a675491904afb002b7e75502b', '76a641d4bb79488a0563dde1041f95c8',
             '2676346b213d6e0f4da6fda09f0b2b92')


def sequences(root, subjects):
    for subject in subjects:
        path = root / f'SUBJECT{subject}.zip'
        if hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest() != CHECKSUMS[subject-1]:
            raise ValueError(f'UP-Fall archive checksum mismatch: {path}')
        with ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith('.csv'):
                    continue
                with TextIOWrapper(archive.open(name), encoding='utf-8-sig') as source:
                    reader = csv.DictReader(source)
                    label_key = next((key for key in ('LABEL', 'LLABEL') if key in reader.fieldnames), None)
                    if label_key is None:
                        continue
                    rows = list(reader)
                impact = next((i for i, row in enumerate(rows) if row[label_key] == '1'), None)
                if not impact:
                    continue
                points = [[[float(row[f'Joint{joint+1}_X'])*500,
                            float(row[f'Joint{joint+1}_Y'])*700, 1.] for joint in JOINTS] for row in rows]
                yield name, impact, points


def features(points, fps):
    history = PoseHistory()
    for index, pose in enumerate(points):
        visible = [p for p in pose if all(math.isfinite(v) for v in p) and 0 < p[0] < 500 and 0 < p[1] < 700]
        if len(visible) < 8:
            continue
        pose = [p if p in visible else [0., 0., 0.] for p in pose]
        x, y = zip(*(p[:2] for p in visible))
        margin = .08*max(max(x)-min(x), max(y)-min(y))
        box = (min(x)-margin, min(y)-margin, max(x)+margin, max(y)+margin)
        _, temporal = history.update(box, pose, index/fps)
        if temporal is not None:
            yield index, temporal


def examples(root):
    rows = []
    count = 0
    for _, impact, points in sequences(root, (1, 2, 3)):
        count += 1
        # The ZIPs omit timestamps. Vary the assumed sampling rate and use
        # only windows close to the first positive impact annotation.
        for fps in (15, 20, 25):
            rows.extend((motion, 1) for i, motion in features(points, fps)
                        if impact <= i <= impact + fps*.25)
    if not rows:
        raise ValueError('No usable UP-Fall impact windows')
    return rows, count


def evaluate_impact(root, model):
    """External check on held-out subjects, against impact annotations only."""
    result = {}
    for fps in (15, 20, 25):
        total = detected = premature = 0
        for _, impact, points in sequences(root, (4, 5)):
            track = LearnedTrack()
            alerts = [i for i, motion in features(points, fps) if track.motion(model, motion, i/fps)[1]]
            total += 1
            detected += any(-.25 <= (i-impact)/fps <= 1. for i in alerts)
            premature += any((i-impact)/fps < -.25 for i in alerts)
        result[fps] = dict(sequences=total, detected=detected, premature=premature)
    return result

"""Small supervised models on the same keypoints produced by YOLO at runtime."""
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).with_name('learned_pose.json')
POSTURES = ('UPRIGHT', 'SITTING', 'LYING')
FEATURE_VERSION = 1


def center(points, indices):
    visible = [points[i][:2] for i in indices if len(points[i]) >= 3
               and all(math.isfinite(v) for v in points[i][:3]) and points[i][2] >= .35
               and any(points[i][:2])]
    return tuple(sum(axis)/len(axis) for axis in zip(*visible)) if visible else None


def direction(first, second):
    if first is None or second is None:
        return (0., 0., 0.)
    dx, dy = second[0]-first[0], second[1]-first[1]
    length = math.hypot(dx, dy)
    return (abs(dx)/length, dy/length, 1.) if length >= 2 else (0., 0., 0.)


@dataclass(frozen=True)
class Pose:
    shoulder: tuple
    hip: tuple
    feet: tuple | None
    height: float
    scale: float
    static: tuple


def describe(box, points):
    if points is None or len(points) != 17:
        return None
    shoulder, hip = center(points, (5, 6)), center(points, (11, 12))
    if shoulder is None or hip is None or math.dist(shoulder, hip) < 4:
        return None
    knee, ankle = center(points, (13, 14)), center(points, (15, 16))
    w, h = max(1., box[2]-box[0]), max(1., box[3]-box[1])
    scale = max(math.dist(shoulder, hip)*2.5, h, w, 12.)
    static = (math.log(w/h), *direction(shoulder, hip), *direction(hip, knee),
              *direction(knee, ankle), *direction(hip, ankle), *direction(shoulder, ankle))
    return Pose(shoulder, hip, ankle, h, scale, static)


class PoseHistory:
    def __init__(self):
        self.samples = deque()

    def update(self, box, points, now):
        while self.samples and now-self.samples[0][0] > .95:
            self.samples.popleft()
        pose = describe(box, points)
        if pose is None:
            return None, None
        recent = [(time, old) for time, old in self.samples if now-time >= .14]
        self.samples.append((now, pose))
        if not recent:
            return pose.static, None
        features = list(pose.static)
        for interval in (.3, .65):
            time, old = min(recent, key=lambda item: abs(now-item[0]-interval))
            dt = now-time
            scale = old.scale
            features.extend((dt, (pose.shoulder[1]-old.shoulder[1])/scale,
                             (pose.hip[1]-old.hip[1])/scale,
                             abs(pose.hip[0]-old.hip[0])/scale,
                             math.log(pose.height/old.height), pose.static[1]-old.static[1],
                             pose.static[0]-old.static[0],
                             (pose.feet[1]-old.feet[1])/scale if pose.feet and old.feet else 0.,
                             float(bool(pose.feet and old.feet))))
        return pose.static, features


class LinearModel:
    def __init__(self, data):
        self.mean = np.asarray(data['mean'])
        self.scale = np.asarray(data['scale'])
        self.weights = np.asarray(data['weights'])

    def probabilities(self, features):
        normalized = np.clip((features-self.mean)/self.scale, -6, 6)
        scores = normalized @ self.weights[:-1] + self.weights[-1]
        values = np.exp(scores-scores.max(axis=-1, keepdims=True))
        return values/values.sum(axis=-1, keepdims=True)


class LearnedPose:
    def __init__(self, data):
        self.posture = LinearModel(data['posture'])
        self.motion = LinearModel(data['motion'])
        self.policy = data['policy']

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['feature_version'] != FEATURE_VERSION:
            raise ValueError('Unsupported learned pose feature version')
        return cls(data)


class LearnedTrack:
    def __init__(self):
        self.history = PoseHistory()
        self.last_risk = None
        self.last_candidate = None
        self.candidate_count = 0

    def posture(self, model, features, observed):
        if features is None:
            return observed
        upright, sitting, lying = model.posture.probabilities(features)
        threshold = model.policy['posture_threshold']
        if observed == 'UNKNOWN' and lying >= threshold and lying > max(upright, sitting):
            return 'LYING'
        if (observed != 'LYING' and features[-1] and features[6]
                and sitting >= threshold and sitting > max(upright, lying)):
            return 'SITTING'
        return observed

    def motion(self, model, features, now):
        score = model.motion.probabilities(features)[1] if features is not None else None
        risk = score is not None and score >= model.policy['motion_threshold']
        confirmation = model.policy.get('motion_confirmation')
        if confirmation is not None:
            if risk:
                if self.last_candidate is None or now-self.last_candidate > confirmation['max_gap']:
                    self.candidate_count = 0
                self.candidate_count += 1
                self.last_candidate = now
                risk = score >= confirmation['immediate'] or self.candidate_count >= confirmation['frames']
            else:
                self.last_candidate = None
                self.candidate_count = 0
        was_active = self.last_risk is not None and now-self.last_risk < 1
        if risk:
            self.last_risk = now
        return bool(risk or was_active), bool(risk and not was_active)

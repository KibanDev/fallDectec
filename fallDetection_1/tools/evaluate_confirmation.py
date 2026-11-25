"""Calibrate motion confirmation on GMDCSA24 subject 3 and replay subject 4."""
import json
import statistics
from types import SimpleNamespace

from train_pose import (ROOT, MODEL_PATH, LearnedPose, LearnedTrack, LinearModel,
                        baseline_engine, cauca_examples, clips, compare, cv2, examples, fit, pose_rows)


def score_policy(dataset, policy):
    model = SimpleNamespace(policy=policy, motion=SimpleNamespace(probabilities=lambda score: (1-score, score)))
    detected = false = 0
    delays = []
    for clip, frames in dataset:
        track = LearnedTrack()
        alerts = [t for t, score in frames if track.motion(model, score, t)[1]]
        if not clip.is_fall:
            false += bool(alerts)
        elif clip.onset is not None:
            valid = [t-clip.onset for t in alerts if -.25 <= t-clip.onset <= 1.5]
            if valid:
                detected += 1
                delays.append(valid[0])
    return dict(detected_falls=detected, adl_clips_alerted=false,
                median_delay_s=statistics.median(delays) if delays else None)


def main():
    cv2.setNumThreads(2)
    reference_path = ROOT/'results'/'confirmation_reference.json'
    weights = json.loads((reference_path if reference_path.is_file() else MODEL_PATH).read_text(encoding='utf-8'))
    model = LearnedPose(weights)
    dataset = list(clips(ROOT/'data'/'gmdcsa'))
    validation_features = [(clip, [(t, motion) for t, _, motion, _ in pose_rows(clip)])
                           for clip in dataset if clip.subject == 3]

    def scores(classifier):
        return [(clip, [(t, float(classifier.probabilities(motion)[1]) if motion is not None else None)
                        for t, motion in rows]) for clip, rows in validation_features]

    training = [clip for clip in dataset if clip.subject in (1, 2)]
    _, motion = examples(training)
    _, auxiliary = cauca_examples(ROOT/'data'/'caucafall'/'keypoints_raw.csv')
    # Increase the weight of labelled everyday movements often mistaken for falls.
    difficult = [(features, 0) for clip in training if not clip.is_fall
                 for t, _, features, _ in pose_rows(clip) if features is not None
                 and any(label in ('exercising', 'sitting', 'sleeping') and start-.35 <= t <= end
                         for label, start, end in clip.intervals)]
    focused = fit(motion + difficult, 2, auxiliary, hard_negatives=model.motion)
    reference_policy = {key: value for key, value in model.policy.items() if key != 'motion_confirmation'}
    reference = score_policy(scores(model.motion), reference_policy)
    candidates = []
    for name, motion_weights in [('reference_weights', weights['motion']), ('focused_training', focused)]:
        validation = scores(LinearModel(motion_weights))
        for frames in (2, 3):
            for threshold in (.4, .45, .5, .55, .6, .625, .65, .7, .75, .8, .85, .9, .95):
                for immediate in (.625, .65, .675, .7, .75, .82, .88, .94, .98, 1.):
                    if immediate <= threshold:
                        continue
                    policy = dict(reference_policy, motion_threshold=threshold,
                                  motion_confirmation=dict(frames=frames, max_gap=.65, immediate=immediate))
                    metrics = score_policy(validation, policy)
                    candidates.append((policy, metrics, name, motion_weights))
    # Explore a trade-off of at most two validation misses. Deployment still
    # requires checking the frozen policy on the separate evaluation clips.
    eligible = [candidate for candidate in candidates
                if candidate[1]['detected_falls'] >= reference['detected_falls']-2]
    if not eligible:
        raise RuntimeError('No confirmation policy retained validation recall')
    policy, metrics, name, weights['motion'] = min(eligible, key=lambda item: (item[1]['adl_clips_alerted'],
                                                               -item[1]['detected_falls'], item[1]['median_delay_s']))
    print('Validation:', reference, metrics, 'Frozen candidate:', policy, flush=True)
    weights['policy'] = policy
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    (out/'confirmation_candidate.json').write_text(json.dumps(weights, separators=(',', ':'))+'\n', encoding='utf-8')
    results, recordings = compare([clip for clip in dataset if clip.subject == 4], LearnedPose(weights),
                                  baseline_engine(ROOT.parent/'fallDetection_0'))
    report = dict(validation_reference=reference, validation_candidate=metrics, policy=policy,
                  selected_weights=name, reweighted_adl_windows=len(difficult),
                  results=results, recordings=recordings)
    (out/'confirmation_gmdcsa.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(results['learned_early'], flush=True)


if __name__ == '__main__':
    main()

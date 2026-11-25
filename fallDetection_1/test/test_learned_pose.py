from types import SimpleNamespace
import unittest

from python_app.fall_fusion import FallFusion
from python_app.learned_pose import LearnedPose, LearnedTrack, PoseHistory


class LearnedPoseTests(unittest.TestCase):
    def test_confirmation_rejects_isolated_scores_and_accepts_strong_scores(self):
        model = SimpleNamespace(policy={'motion_threshold': .6, 'motion_confirmation':
                                        {'frames': 2, 'max_gap': .65, 'immediate': .9}},
                                motion=SimpleNamespace(probabilities=lambda score: (1-score, score)))
        track = LearnedTrack()
        self.assertEqual(track.motion(model, .7, .1), (False, False))
        self.assertEqual(track.motion(model, None, .2), (False, False))
        self.assertEqual(track.motion(model, .7, .3), (False, False))
        self.assertEqual(track.motion(model, .7, 1.), (False, False))
        self.assertEqual(track.motion(model, .7, 1.1), (True, True))
        self.assertEqual(LearnedTrack().motion(model, .95, 0.), (True, True))

    def test_weights_match_live_feature_sizes(self):
        model = LearnedPose.load()
        history = PoseHistory()
        points = [[0, 0, 0] for _ in range(17)]
        for index, point in ((5, [40, 20, .9]), (11, [40, 60, .9]),
                             (13, [65, 65, .9]), (15, [70, 90, .9])):
            points[index] = point
        static, temporal = history.update([25, 10, 85, 105], points, 0)
        self.assertIsNone(temporal)
        static, temporal = history.update([25, 10, 85, 105], points, .2)
        self.assertAlmostEqual(sum(model.posture.probabilities(static)), 1)
        self.assertAlmostEqual(sum(model.motion.probabilities(temporal)), 1)
        _, temporal = history.update([25, 10, 85, 105], points, 2.)
        self.assertIsNone(temporal)

    def test_motion_alerts_once_per_track_and_requires_pose_history(self):
        model = SimpleNamespace(policy={'motion_threshold': .7},
                                motion=SimpleNamespace(probabilities=lambda _: (.1, .9)))
        one, two = LearnedTrack(), LearnedTrack()
        self.assertEqual(one.motion(model, None, .1), (False, False))
        self.assertEqual(one.motion(model, [1], .2), (True, True))
        self.assertEqual(one.motion(model, [1], .4), (True, False))
        self.assertEqual(two.motion(model, [1], .4), (True, True))
        self.assertEqual(one.motion(model, None, 1.5), (False, False))
        self.assertEqual(one.motion(model, [1], 1.6), (True, True))

    def test_sitting_is_normal_but_does_not_clear_a_confirmed_fall(self):
        fusion = FallFusion()
        for index in range(5):
            ordinary = fusion.update('SITTING', 'NORMAL', index*.5)
            self.assertFalse(ordinary.alert or ordinary.warning)
        self.assertEqual(ordinary.status, 'NORMAL')
        for index in range(5):
            fall = fusion.update('LYING', 'IMPACT', 2.5+index*.5)
        self.assertEqual(fall.status, 'FALL_DETECTED')
        for index in range(7):
            seated = fusion.update('SITTING', 'NORMAL', 5+index*.5)
        self.assertEqual(seated.status, 'FALL_DETECTED')

from unittest import TestCase

import numpy as np

from evaluation import segment_iou


class TestEvaluation(TestCase):
    def test_segment_iou(self):
        target_segment = np.array([10, 20])
        candidate_segments1 = np.array([[10, 20]])
        candidate_segments2 = np.array([[10, 20], [10, 18]])

        np.testing.assert_array_equal(segment_iou(target_segment, candidate_segments1), 1)
        np.testing.assert_array_equal(segment_iou(target_segment, candidate_segments2), [1, 0.8])

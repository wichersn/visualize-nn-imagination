import unittest

from train import visualize_metric
from train.train import CountAccuracyInverseMetric
from matplotlib import pyplot as plt
import numpy as np
from train.data_functions import num_black_cells, gen_data_batch, get_batch

from absl import app
from absl import flags
FLAGS = flags.FLAGS

non_train_indexies = [1,2,3]

class MetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])
        self.metric_test_eval_datas = gen_data_batch(2000, 4)
        FLAGS.game_timesteps = 4
        FLAGS.model_timesteps = 4

    def metric_asserts(self, eval_datas, gen_boards, thresh, expected_min, expected_max):
        metric_val = visualize_metric.visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies)

        self.assertGreaterEqual(metric_val, expected_min)
        self.assertLessEqual(metric_val, expected_max)

    def test_same_data_should_be_1_even_when_reordered(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 4]], axis=1)
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .99, 1, 1.1)

    def test_threshold_works(self):
        self.assertLess(
            visualize_metric.visualize_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, .99, non_train_indexies),
            visualize_metric.visualize_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, .4, non_train_indexies))

    def test_only_gets_partial_credit_if_repeating_same_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 2],
             self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 4]], axis=1)
        # It should get (1 + 1 + .5) / 3 = .83
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .99, .83, .88)

    def test_no_credit_for_start_or_end_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        # It should get (1 + 0 + 0) / 3 = .33
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .99, .33, .4)

    def test_combine_metric_works(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        combine_val = visualize_metric.combine_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, metric_test_gen_datas, .95, non_train_indexies)
        print(combine_val)
        self.assertGreaterEqual(combine_val, 1)
        self.assertLessEqual(combine_val, 1.1)

class CountAccuracyInverseMetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])

    def test(self):
      metric = CountAccuracyInverseMetric()
      y_pred = np.array([21.3, 10.6, 70.8, 90, 35.2]) / (FLAGS.board_size **2)
      y_true = np.array([21, 10, 70, 97, 35]) / (FLAGS.board_size **2)
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(1-2/5, metric.result().numpy())

if __name__ == '__main__':
    unittest.main()

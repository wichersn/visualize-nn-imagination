import unittest

from train import visualize_metric
from train.train import AccuracyInverseMetric
import numpy as np
from train.data_functions import num_black_cells, gen_data_batch, get_batch

from absl import flags
FLAGS = flags.FLAGS

non_train_indexies = [1,2,3]

class MetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])
        self.metric_test_eval_datas = gen_data_batch(2000, 4)

    def metric_asserts(self, eval_datas, gen_boards, expected_min, expected_max):
        metric_val = visualize_metric.visualize_metric(eval_datas, gen_boards, non_train_indexies)

        self.assertGreaterEqual(metric_val, expected_min)
        self.assertLessEqual(metric_val, expected_max)

    def test_same_data_should_be_1_even_when_reordered(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 4]], axis=1)
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, 1, 1.1)

    def test_only_gets_partial_credit_if_repeating_same_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 2],
             self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 4]], axis=1)
        # It should get less than 1 since 2 states are the same
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .65, .90)

    def test_no_credit_for_start_or_end_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        # It should get close to 0
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .0, .4)

    def test_combine_metric_works(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        combine_val = visualize_metric.combine_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, metric_test_gen_datas, non_train_indexies)
        print(combine_val)
        self.assertGreaterEqual(combine_val, 1)
        self.assertLessEqual(combine_val, 1.1)

class AccuracyInverseMetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])

    def test(self):
      metric = AccuracyInverseMetric(FLAGS.board_size)
      y_pred = np.array([21.3, 10.6, 70.8, 90, 35.2]) / (FLAGS.board_size **2)
      y_true = np.array([21, 10, 70, 97, 35]) / (FLAGS.board_size **2)
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(1-2/5, metric.result().numpy())

if __name__ == '__main__':
    unittest.main()

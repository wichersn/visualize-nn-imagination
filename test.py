import unittest
import train
from matplotlib import pyplot as plt
import numpy as np

from absl import app
from absl import flags
FLAGS = flags.FLAGS

non_train_indexies = [1,2,3]

class TestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])
        self.metric_test_eval_datas = train.gen_data_batch(2000, 4)

    def metric_asserts(self, eval_datas, gen_boards, thresh, expected_min, expected_max):
        metric_val = train.metric(eval_datas, gen_boards, thresh, non_train_indexies)

        self.assertGreaterEqual(metric_val, expected_min)
        self.assertLessEqual(metric_val, expected_max)

    def test_same_data_should_be_1_even_when_reordered(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 4]], axis=1)
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .99, 1, 1.1)

    def test_threshold_works(self):
        self.assertLess(train.metric(self.metric_test_eval_datas, self.metric_test_eval_datas, .99, non_train_indexies),
                        train.metric(self.metric_test_eval_datas, self.metric_test_eval_datas, .4, non_train_indexies))

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
        combine_val = train.combine_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, metric_test_gen_datas, .95, non_train_indexies)
        print(combine_val)
        self.assertGreaterEqual(combine_val, 1)
        self.assertLessEqual(combine_val, 1.1)


if __name__ == '__main__':
    unittest.main()

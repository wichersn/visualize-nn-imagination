# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from train import visualize_metric
from train.train import AccuracyInverseMetric
import numpy as np
from train.data_functions import num_black_cells, gen_data_batch, get_batch


from absl import flags
FLAGS = flags.FLAGS
FLAGS.random_board_prob = 1.0

non_train_indexies = [1,2,3]

class MetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])
        self.metric_test_eval_datas = gen_data_batch(500, 4)

    def metric_asserts(self, eval_datas, gen_boards, expected_min, expected_max, non_train_indexies):
        metric_val = visualize_metric.visualize_metric(eval_datas, gen_boards, non_train_indexies)

        self.assertGreaterEqual(metric_val, expected_min)
        self.assertLessEqual(metric_val, expected_max)

    def test_same_data_should_be_1_even_when_reordered(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 4]], axis=1)
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .999, 1.01, non_train_indexies)

    def test_no_credit_repeating_same_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 2],
             self.metric_test_eval_datas[:, 1], self.metric_test_eval_datas[:, 4]], axis=1)
        # (1+1+0) / 3
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .666, .8, non_train_indexies)

    def test_repeating_first_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0]], axis=1)
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .0, .07, non_train_indexies)

    def test_all_zeros(self):
        self.metric_asserts(np.zeros_like(self.metric_test_eval_datas), np.zeros_like(self.metric_test_eval_datas), .999, 1.01, non_train_indexies)

    def test_all_ones(self):
        self.metric_asserts(np.ones_like(self.metric_test_eval_datas), np.ones_like(self.metric_test_eval_datas), .999, 1.01, non_train_indexies)

    def test_gen_zeros(self):
        self.metric_asserts(np.zeros_like(self.metric_test_eval_datas), self.metric_test_eval_datas, .0, .01, non_train_indexies)

    def test_no_credit_for_start_or_end_state(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        # It should be around 0
        self.metric_asserts(self.metric_test_eval_datas, metric_test_gen_datas, .0, .3, non_train_indexies)

    def test_combine_metric_works(self):
        metric_test_gen_datas = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        combine_val = visualize_metric.combine_metric(self.metric_test_eval_datas, self.metric_test_eval_datas, metric_test_gen_datas, non_train_indexies)
        print(combine_val)
        self.assertGreaterEqual(combine_val, 1)
        self.assertLessEqual(combine_val, 1.1)

    def test_game3_model4_steps(self):
        gen_boards = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 1],
             self.metric_test_eval_datas[:, 1],
             self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3]], axis=1)
        eval_datas = self.metric_test_eval_datas[:, :4]
        self.metric_asserts(eval_datas, gen_boards, .99, 1.01, non_train_indexies)

    def test_game3_model4_steps2(self):
        gen_boards = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 0],
             self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3]], axis=1)
        eval_datas = self.metric_test_eval_datas[:, :4]
        # expected score: (1+0) / 2 = .5
        self.metric_asserts(eval_datas, gen_boards, .5, .67, non_train_indexies)

    def test_game4_model3_steps(self):
        gen_boards = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 1],
             self.metric_test_eval_datas[:, 2], self.metric_test_eval_datas[:, 3]], axis=1)
        eval_datas = self.metric_test_eval_datas
        # expected score: (1+1+0) / 2 = 1.0
        self.metric_asserts(eval_datas, gen_boards, 1.0, 1.2, [1,2])

    def test_game4_model3_steps_2(self):
        gen_boards = np.stack(
            [self.metric_test_eval_datas[:, 0], self.metric_test_eval_datas[:, 3],
             self.metric_test_eval_datas[:, 4], self.metric_test_eval_datas[:, 4]], axis=1)
        eval_datas = self.metric_test_eval_datas
        # expected score: (0+1+0) / 2 = .5
        self.metric_asserts(eval_datas, gen_boards, .54, .8, [1,2])

    def test_game3_model2_steps(self):
        gen_boards = np.stack(
            [self.metric_test_eval_datas[:, 0], np.zeros_like(self.metric_test_eval_datas[:, 4]), self.metric_test_eval_datas[:, 4]], axis=1)
        eval_datas = self.metric_test_eval_datas[:, :4]
        # expected score: (0+0) / 1 = .0
        self.metric_asserts(eval_datas, gen_boards, .0, .37, [1])

class AccuracyInverseMetricTestCase(unittest.TestCase):
    def setUp(self):
        FLAGS(['test'])

    def test_board_size(self):
      metric = AccuracyInverseMetric(FLAGS.board_size)
      y_pred = np.array([21.3, 10.6, 70.8, 90, 35.2]) / (FLAGS.board_size **2)
      y_true = np.array([21, 10, 70, 97, 35]) / (FLAGS.board_size **2)
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(1-2/5, metric.result().numpy())

    def test_size_2(self):
      metric = AccuracyInverseMetric(2)
      y_pred = np.array([1.49, .51, 1.51, 3.4, 3.4]) / 4.0
      y_true = np.array([1,     0,  2,    4,   3])  / 4.0
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(1-3/5, metric.result().numpy())

if __name__ == '__main__':
    unittest.main()

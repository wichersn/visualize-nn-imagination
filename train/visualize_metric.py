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

import numpy as np
from sklearn.metrics import matthews_corrcoef


def single_index_metric(gt, gen):
  """Returns the average absolute value of the matthews correlation coefficient(MCC) between generated and true states
  0 is no correlation, 1 is perfect
  """
  if (gt == gen).all():
    # matthews_corrcoef will return .5 if both gt and gen are all 0s or 1s.
    return 1.0

  mccs = matthews_corrcoef(gt.flatten(), gen.flatten() > .5)
  return max(mccs, 0)


def single_gt_index_metric(gt, gen_boards, non_train_indexies):
  """Metric representing how well the gen boards represent the single ground truth state.

  It gets credit for only the generated state it matches the best.
  """
  metrics_for_gt = []
  for j in non_train_indexies:
    metrics_for_gt.append(single_index_metric(gt, gen_boards[:, j]))
  return np.max(metrics_for_gt)  

def visualize_metric(eval_datas, gen_boards, non_train_indexies):
  """Averages the metric on each of the ground truth states."""
  train_indexies = np.array(list(set(range(gen_boards.shape[1])) - set(non_train_indexies)))
  game_train_indexes = train_indexies * eval_datas.shape[1] // gen_boards.shape[1]
  game_non_train_indexes = set(range(eval_datas.shape[1])) - set(game_train_indexes)
  total_metric = 0
  for game_i in game_non_train_indexes:
    total_metric += single_gt_index_metric(eval_datas[:, game_i], gen_boards, non_train_indexies)
  return total_metric / float(min(len(non_train_indexies), len(game_non_train_indexes)))

def combine_metric(eval_datas, gen_boards, adver_gen_boards, non_train_indexies):
  adver_metric = visualize_metric(eval_datas, adver_gen_boards, non_train_indexies)
  regular_metric = visualize_metric(eval_datas, gen_boards, non_train_indexies)
  return max(adver_metric, regular_metric)

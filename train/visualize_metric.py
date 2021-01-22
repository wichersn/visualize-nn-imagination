import numpy as np
from absl import flags
FLAGS = flags.FLAGS

def single_index_metric(gt, gen, thresh):
  """Returns the porportion of gen instances close enough to the ground truth."""
  equal = np.equal(gt, gen>.5)
  acc = np.mean(equal, (1,2,3))
  over_thresh = acc >= thresh
  return np.mean(over_thresh)


def single_gt_index_metric(gt, gen_boards, thresh, non_train_indexies):
  """Metric representing how well the gen boards represent the single ground truth state.

  It gets full credit the time it predicts the state the best, and partial credit for all other times.
  """
  metrics_for_gt = []
  for j in non_train_indexies:
    metrics_for_gt.append(single_index_metric(gt, gen_boards[:, j], thresh))
  metrics_for_gt.sort(reverse = True)
  weights = np.ones([len(non_train_indexies)]) * .5
  weights[0] = 1
  result = sum(np.multiply(metrics_for_gt, weights))
  return result


def visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies):
  """Averages the metric on each of the ground truth states."""
  train_indexies = np.array(list(set(range(gen_boards.shape[1])) - set(non_train_indexies)))
  game_train_indexes = train_indexies * eval_datas.shape[1] // gen_boards.shape[1]
  game_non_train_indexes = set(range(eval_datas.shape[1])) - set(game_train_indexes)
  total_metric = 0
  for game_i in game_non_train_indexes:
    total_metric += single_gt_index_metric(eval_datas[:, game_i], gen_boards, thresh, non_train_indexies)

  return total_metric / float(min(len(non_train_indexies), len(game_non_train_indexes)))


def combine_metric(eval_datas, gen_boards, adver_gen_boards, thresh, non_train_indexies):
  adver_metric = visualize_metric(eval_datas, adver_gen_boards, thresh, non_train_indexies)
  regular_metric = visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies)
  return max(adver_metric, regular_metric)
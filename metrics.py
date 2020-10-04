import numpy as np


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


def metric(eval_datas, gen_boards, thresh, non_train_indexies):
  """Averages the metric on each of the ground truth states."""
  total_metric = 0
  for i in non_train_indexies:
    total_metric += single_gt_index_metric(eval_datas[:, i], gen_boards, thresh, non_train_indexies)
  return total_metric / float(len(non_train_indexies))


def combine_metric(eval_datas, gen_boards, adver_gen_boards, thresh, non_train_indexies):
  adver_metric = metric(eval_datas, adver_gen_boards, thresh, non_train_indexies)
  regular_metric = metric(eval_datas, gen_boards, thresh, non_train_indexies)
  return max(adver_metric, regular_metric)
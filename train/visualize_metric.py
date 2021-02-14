import numpy as np
from sklearn.metrics import matthews_corrcoef


def single_index_metric(gt, gen):
  """Returns the average absolute value of the matthews correlation coefficient(MCC) between generated and true states
  0 is no correlation, 1 is perfect
  """
  mccs = np.ones(len(gen))
  for i in range(len(gen)):
    mcc = matthews_corrcoef(gt[i].flatten(), gen[i].flatten()>.5)
    mccs[i] = abs(mcc)

  return np.mean(mccs)


def single_gen_index_metric(gt_boards, gen, non_train_indexies):
  """Metric representing how well a single generated state matches the true states.
  Only counts the true states that match the best
  """
  metrics_for_gt = []
  for j in non_train_indexies:
    metrics_for_gt.append(single_index_metric(gt_boards[:, j], gen))
  return max(metrics_for_gt), np.argmax(metrics_for_gt)


def visualize_metric(eval_datas, gen_boards, non_train_indexies):
  """Averages the metric on each of the generated states. Each generated states is matched to a unique true state"""
  total_metric = 0
  used_idxs = []
  for i in non_train_indexies:
    metric_val, idx = single_gen_index_metric(eval_datas, gen_boards[:, i], non_train_indexies)
    if idx not in used_idxs:
      total_metric += metric_val
      used_idxs.append(idx)

  return total_metric / float(len(non_train_indexies))


def combine_metric(eval_datas, gen_boards, adver_gen_boards, non_train_indexies):
  adver_metric = visualize_metric(eval_datas, adver_gen_boards, non_train_indexies)
  regular_metric = visualize_metric(eval_datas, gen_boards, non_train_indexies)
  return max(adver_metric, regular_metric)
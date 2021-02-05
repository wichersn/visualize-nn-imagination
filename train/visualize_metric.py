import numpy as np
from sklearn.metrics import matthews_corrcoef


def single_index_metric(gt, gen):
  """Returns the average absolute value of the matthews correlation coefficient(MCC) between generated and true states
  0 is no correlation, 1 is perfect
  """
  mccs = np.ones(len(gen))
  print(gt.shape, gen.shape)
  for i in range(len(gen)):
    mcc = matthews_corrcoef(gt[i].flatten(), gen[i].flatten()>.5)
    mccs[i] = abs(mcc)

  return np.mean(mccs)


def single_gen_index_metric(gt_boards, gen, non_train_indexies):
  """Metric representing how well a single generated state matches the true states.
  Only counts the closest match
  """
  metrics_for_gt = []
  for j in non_train_indexies:
    metrics_for_gt.append(single_index_metric(gt_boards[:, j], gen))
  return max(metrics_for_gt)


def visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies):
  """Averages the metric on each of the generated states."""
  total_metric = 0
  for i in non_train_indexies:
    total_metric += single_gen_index_metric(eval_datas, gen_boards[:, i], non_train_indexies)
  return total_metric / float(len(non_train_indexies))


def combine_metric(eval_datas, gen_boards, adver_gen_boards, thresh, non_train_indexies):
  adver_metric = visualize_metric(eval_datas, adver_gen_boards, thresh, non_train_indexies)
  regular_metric = visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies)
  return max(adver_metric, regular_metric)
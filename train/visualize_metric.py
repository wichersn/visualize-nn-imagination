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
  total_metric = 0
  for i in non_train_indexies:
    total_metric += single_gt_index_metric(eval_datas[:, i], gen_boards, non_train_indexies)
  return total_metric / float(len(non_train_indexies))

def combine_metric(eval_datas, gen_boards, adver_gen_boards, non_train_indexies):
  adver_metric = visualize_metric(eval_datas, adver_gen_boards, non_train_indexies)
  regular_metric = visualize_metric(eval_datas, gen_boards, non_train_indexies)
  return max(adver_metric, regular_metric)
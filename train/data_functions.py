import numpy as np
from scipy.signal import convolve2d
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_multi_integer('board_size', [20,20], '')

def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def num_black_cells(X):
    return np.sum(X, axis=(1, 2))


def convert_model_in(data):
  data = np.array(data)
  data = data.astype(np.float32)
  data = np.expand_dims(data, -1)
  return data


def gen_data_batch(size, skip):
  datas = []
  start_next = None
  for _ in range(size):
    if np.random.rand(1) < .1 or start_next is None:
      life_state = np.random.rand(FLAGS.board_size[0], FLAGS.board_size[1]) > .5
    else:
      life_state = start_next

    data = []
    data.append(life_state)
    for i in range(skip):
      life_state = life_step(life_state)
      data.append(life_state)
      if i == 0:
        start_next = life_state
    datas.append(data)

  datas = convert_model_in(datas)

  return datas


def get_batch(datas, targets, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    return datas[idx], targets[idx]
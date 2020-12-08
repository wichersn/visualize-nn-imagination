import numpy as np
from scipy.signal import convolve2d
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 20, '')
flags.DEFINE_float('random_board_prob', .1, '')

def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def num_black_cells(X):
    return np.sum(X, axis=(-1, -2, -3))


def convert_model_in(data):
  data = np.array(data)
  data = data.astype(np.float32)
  data = np.expand_dims(data, -1)
  return data


def gen_data_batch(size, skip):
  datas = []
  start_next = None
  for _ in range(size):
    if np.random.rand(1) < FLAGS.random_board_prob or start_next is None:
      life_state = np.random.rand(FLAGS.board_size, FLAGS.board_size) > .5
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

def plt_boards(boards):
  """Use this function in get_batch to debug. Ex: plt_boards(datas[idx][0]), plt_boards(np.array([targets[idx][0]]))"""
  import matplotlib.pyplot as plt

  f = plt.figure()
  for i in range(len(boards)):
    # print("batch_targets", batch_targets[i])
    f.add_subplot(1,len(boards), i+1)
    plt.imshow(boards[i,:,:,0], interpolation='nearest', cmap=plt.cm.binary)
    plt.axis('off')
  plt.show()

def get_batch(datas, targets, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    return datas[idx], targets[idx]
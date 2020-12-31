import numpy as np
from scipy.signal import convolve2d
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 20, '')
flags.DEFINE_float('random_board_prob', .1, '')

def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def num_black_cells(X):
  val = tf.expand_dims(tf.reduce_sum(X, axis=(-1, -2, -3)), 2)
  return val / (FLAGS.board_size ** 2)


def convert_model_in(data):
  data = np.array(data)
  data = data.astype(np.float32)
  data = np.expand_dims(data, -1)
  return data


def gen_data_batch(batch_size, timesteps, alive_cells_size):
  datas = []
  alive_part = np.random.rand(batch_size, alive_cells_size, alive_cells_size) > .5
  life_states = np.zeros([batch_size, FLAGS.board_size, FLAGS.board_size], dtype=np.int32)
  life_states[:, :alive_cells_size, :alive_cells_size] = alive_part

  for _ in range(batch_size):
    life_state = life_states[0]

    data = []
    data.append(life_state)
    for i in range(timesteps):
      life_state = life_step(life_state)
      data.append(life_state)
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

def get_batch(datas, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    return datas[idx]
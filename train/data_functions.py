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
from scipy.signal import convolve2d
from absl import flags
import tensorflow as tf
import matplotlib.pyplot as plt
import io

FLAGS = flags.FLAGS
flags.DEFINE_integer('board_size', 20, '')
flags.DEFINE_float('random_board_prob', .1, '')

def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def num_black_cells(X):
  # X is [batch_size, timesteps, board_size, board_size, 1] and val is [batch_size, timesteps, 1]
  val = tf.expand_dims(tf.reduce_sum(X, axis=(-1, -2, -3)), 2)
  return val / (FLAGS.board_size ** 2)

def num_black_cells_in_patch(X):
  # X is [batch_size, timesteps, board_size, board_size, 1] and targets is same dims as X
  targets = tf.nn.pool(X, (1, FLAGS.patch_size, FLAGS.patch_size), 'AVG', padding='SAME')
  return targets

def num_black_cells_in_grid(X):
  # X is [batch_size, timesteps, board_size, board_size, 1] and targets is same dims as X
  if FLAGS.board_size % FLAGS.grid_size > 0:
  	raise Exception('Board size is not divisible by grid size')
  targets = tf.nn.pool(X, (1, FLAGS.grid_size, FLAGS.grid_size), 'AVG', (1, FLAGS.grid_size, FLAGS.grid_size), padding='SAME')
  # targets is dim ceil(board size / grid size)
  shape = targets.shape
  targets = tf.reshape(targets, [shape[0] * shape[1]] + shape[2:]) # resize expects 1 batch dimension
  targets = tf.image.resize(targets, [FLAGS.board_size, FLAGS.board_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  targets = tf.reshape(targets, shape[:2] + targets.shape[1:])
  return targets


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

def get_batch(datas, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    return datas[idx]

def plt_boards(boards, axes, pos):
  for i in range(len(boards)):
    axes[pos, i].imshow(boards[i,:,:,0], interpolation='nearest', cmap=plt.cm.binary, vmin=0, vmax=1)
  for i in range(axes.shape[1]):
    axes[pos, i].axis('off')

def plt_data(datas):
    # datas: {"gt": [num_display_imgs, game_timesteps, ...], "p": [num_display_imgs, model_timesteps, ...]}
    max_timesteps = 0
    for data in datas.values():
        max_timesteps = max(data.shape[1], max_timesteps)

    one_datas = list(datas.values())[0]
    figs = []
    for i in range(len(one_datas)):
        fig, axes = plt.subplots(len(datas), max_timesteps)
        for pos, name in enumerate(datas):
            plt_boards(datas[name][i], axes, pos)
        figs.append(fig)
    return figs

def fig_to_image(fig):
  io_buf = io.BytesIO()
  fig.savefig(io_buf, format='png')
  io_buf.seek(0)
  image = tf.image.decode_png(io_buf.getvalue(), channels=4)
  io_buf.close()
  return image

def plt_boards_debug(boards):
  """Use this function in get_batch to debug. Ex: plt_boards(datas[idx][0]), plt_boards(np.array([targets[idx][0]]))"""
  import matplotlib.pyplot as plt

  f = plt.figure()
  for i in range(len(boards)):
    # print("batch_targets", batch_targets[i])
    f.add_subplot(1,len(boards), i+1)
    plt.imshow(boards[i,:,:,0], interpolation='nearest', cmap=plt.cm.binary)
    plt.axis('off')
  plt.show()
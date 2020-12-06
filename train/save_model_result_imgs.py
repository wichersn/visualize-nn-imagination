import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from skimage.io import imsave
import os
from tensorflow import io
import tensorflow as tf
from absl import app
import random
from data_functions import num_black_cells

FLAGS = flags.FLAGS

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')
flags.DEFINE_bool('is_hp_serach_root', False, '')

def plt_boards(boards, axes, pos):
  for i in range(len(boards)):
    axes[pos, i].imshow(boards[i,:,:,0], interpolation='nearest', cmap=plt.cm.binary)
    axes[pos, i].axis('off')

def save_imgs(saved_array_dir, ouput_dir, input_array_names, num_to_save):
    datas = {}

    for name in input_array_names:
        input_array_path = os.path.join(saved_array_dir, name)
        datas[name] = np.load(io.gfile.GFile(input_array_path, 'rb'))

    task_gen = None
    try:
        input_array_path = os.path.join(saved_array_dir, "task_gen")
        task_gen = np.load(io.gfile.GFile(input_array_path, 'rb'))
    except tf.python.framework.errors_impl.NotFoundError:
        pass

    for _ in range(num_to_save):
        i = random.randint(0, len(datas[name])-1)
        fig, axes = plt.subplots(len(input_array_names), len(datas[name][i]))
        for pos, name in enumerate(input_array_names):
            plt_boards(datas[name][i], axes, pos)

        # if task_gen is not None:
        #     task_gt = num_black_cells(datas["eval_datas"][i])[-1][0]
        #     error = task_gen[i, -1] - task_gt
        #     file_name = "{}_error{:.4f}.png".format(i, error)
        # else:
        #     file_name = "{}.png".format(i)
        file_name = "{}.png".format(i)
        img_file = io.gfile.GFile(os.path.join(ouput_dir, file_name), 'wb')
        plt.savefig(img_file)
    plt.close('all')

def main(_):
    input_array_names = {"eval_datas": "Ground Truth",
               "gen_boards": "Inferred",
               "adver_gen_boards": "Inferred with adver loss"}

    if FLAGS.is_hp_serach_root:
        for i in range(1, 999):
            try:
                save_imgs(os.path.join(FLAGS.job_dir, str(i)), os.path.join(FLAGS.job_dir, "imgs", str(i)),
                          input_array_names, 20)
            except tf.python.framework.errors_impl.NotFoundError:
                print("No eval for", i)
    else:
        save_imgs(FLAGS.job_dir, os.path.join(FLAGS.job_dir, "imgs"), input_array_names, 20)


if __name__ == '__main__':
  app.run(main)
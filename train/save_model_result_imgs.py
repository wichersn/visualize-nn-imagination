import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from skimage.io import imsave
import os
from tensorflow import io
import tensorflow as tf
from absl import app

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

    for i in range(num_to_save):
        fig, axes = plt.subplots(len(input_array_names), len(datas[name][i]))
        for pos, name in enumerate(input_array_names):
            plt_boards(datas[name][i], axes, pos)
        img_file = io.gfile.GFile(os.path.join(ouput_dir, str(i)+".png"), 'wb')
        plt.savefig(img_file)
    plt.close('all')

def main(_):
    input_array_names = {"eval_datas": "Ground Truth",
               "gen_boards": "Inferred",
               "adver_gen_boards": "Inferred with adver loss"}

    if FLAGS.is_hp_serach_root:
        for i in range(1, 99999):
            try:
                save_imgs(os.path.join(FLAGS.job_dir, str(i)), os.path.join(FLAGS.job_dir, "imgs", str(i)),
                          input_array_names, 20)
            except tf.python.framework.errors_impl.NotFoundError:
                print("No eval for", i)
                pass
    else:
        save_imgs(FLAGS.job_dir, os.path.join(FLAGS.job_dir, "imgs"), input_array_names, 20)


if __name__ == '__main__':
  app.run(main)
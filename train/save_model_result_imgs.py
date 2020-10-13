import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from skimage.io import imsave
import os
from tensorflow import io
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')

def plt_boards(boards, axes, pos):
  for i in range(len(boards)):
    axes[pos, i].imshow(boards[i,:,:,0], interpolation='nearest', cmap=plt.cm.binary)
    axes[pos, i].axis('off')

def save_imgs(input_array_names, num_to_save):
    datas = {}
    for name in input_array_names:
        input_array_path = os.path.join(FLAGS.job_dir, name)
        datas[name] = np.load(io.gfile.GFile(input_array_path, 'rb'))

    for i in range(num_to_save):
        fig, axes = plt.subplots(len(input_array_names), len(datas[name][i]))
        for pos, name in enumerate(input_array_names):
            plt_boards(datas[name][i], axes, pos)
        img_file = io.gfile.GFile(os.path.join(FLAGS.job_dir, "imgs", str(i)+".png"), 'wb')
        plt.savefig(img_file)

def main(_):
    save_imgs({"eval_datas": "Ground Truth",
               "gen_boards": "Inferred",
               "adver_gen_boards": "Inferred with adver loss"},
              20
              )


if __name__ == '__main__':
  app.run(main)
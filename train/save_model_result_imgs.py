import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from skimage.io import imsave
import os
from tensorflow import io
import tensorflow as tf
from absl import app
from train.data_functions import plt_data


FLAGS = flags.FLAGS

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')
flags.DEFINE_bool('is_hp_serach_root', False, '')

def save_imgs(saved_array_dir, ouput_dir, input_array_names, num_to_save):
    datas = {}

    for name in input_array_names:
        input_array_path = os.path.join(saved_array_dir, name)
        try:
            datas[name] = np.load(io.gfile.GFile(input_array_path, 'rb'))
        except tf.errors.NotFoundError:
            print(name, "not in saved arrays", flush=True)
            continue

    if len(datas) > 0:
        one_datas = list(datas.values())[0]
        idx = np.random.choice(np.arange(len(one_datas)), num_to_save, replace=False)
        for key in datas:
            datas[key] = datas[key][idx]
        figs = plt_data(datas)

        for i in range(len(figs)):
            file_name = "{}.png".format(i)
            img_file = io.gfile.GFile(os.path.join(ouput_dir, file_name), 'wb')
            figs[i].savefig(img_file)
        plt.close('all')

def main(_):
    input_array_names = {"eval_datas": "Ground Truth",
               "gen_boards": "Inferred",
               "adver_gen_boards": "Inferred with adver loss",
                         "gen_boards_first_last": "Trained first, last.",
                         "gen_boards_all": "Trained all.",
                         }

    for dec_ts in range(5):
        input_array_names["gen_boards_{}".format(dec_ts)] = "Trained {}".format(dec_ts)

    if FLAGS.is_hp_serach_root:
        for i in range(1, 999):
            try:
                save_imgs(os.path.join(FLAGS.job_dir, str(i)), os.path.join(FLAGS.job_dir, "imgs", str(i)),
                          input_array_names, 20)
            except tf.errors.NotFoundError:
                print("No eval for", i)
    else:
        save_imgs(FLAGS.job_dir, os.path.join(FLAGS.job_dir, "imgs"), input_array_names, 20)


if __name__ == '__main__':
  app.run(main)
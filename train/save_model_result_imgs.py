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

def save_imgs(input_array_name):
    input_array_path = os.path.join(FLAGS.job_dir, input_array_name)

    data = np.load(io.gfile.GFile(input_array_path, 'rb'))
    print("data.shape", data.shape)

    # TODO: Should save all of the images
    imsave(input_array_path+".png", data[0,0,:,:,0])

def main(_):
    save_imgs("eval_datas")
    save_imgs("gen_boards")
    save_imgs("adver_gen_boards")


if __name__ == '__main__':
  app.run(main)
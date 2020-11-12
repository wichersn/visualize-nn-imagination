import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf
from absl import app
from absl import flags
import os

import train.visualize_metric
from train.train import *

def main(_):
  global acc_metrics, metrics
  acc_metrics = [tf.keras.metrics.BinaryAccuracy() for _ in range(FLAGS.num_timesteps+1)]
  for i in range(FLAGS.num_timesteps+1):
    metrics.append(["acc at {}".format(i), acc_metrics[i]])

  datas = gen_data_batch(100000, FLAGS.num_timesteps)
  eval_datas = gen_data_batch(FLAGS.eval_data_size, FLAGS.num_timesteps)
  encoder, intermediates, decoder, decoder_counter, model, discriminator = create_models(
    FLAGS.encoded_size, FLAGS.num_timesteps, FLAGS.use_residual)

  optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate)
  # TODO: Lower learning rate as non train accuracy improves. Might help not mess up the hidden representations that it learned.
  discriminator_opt=tf.keras.optimizers.Adam()

  train_indexies = [0]
  non_train_indexies = range(0, FLAGS.num_timesteps)
  target_train_mse = 0.3
  print("Full model training")
  train_mse = get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, acc_metrics, reg_amount=FLAGS.reg_amount)(
    decoder, decoder_counter, train_indexies, [], True, .99, target_train_mse, "count_cells")

  if train_mse < target_train_mse:
    save_metric_result(1 - train_mse, "final_metric_result")
    return

  adver_decoder = tf.keras.Sequential(
      [
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x)),
      ], name="adver_decoder"
  )
  for _ in range(FLAGS.decoder_layers-1):
    adver_decoder.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  adver_decoder.add(tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("adver_decoder", adver_decoder.layers)

  print("Training Only Decoder")
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, acc_metrics, reg_amount=FLAGS.reg_amount)(
    adver_decoder, decoder_counter, train_indexies, [], False, .96, -1, "train_decoder")

  print("Training Only Decoder Adversarial")
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, acc_metrics, reg_amount=FLAGS.reg_amount)(
    adver_decoder, decoder_counter, train_indexies, non_train_indexies, False, .98, -1, "train_decoder_adversarial")

  model_results = model(eval_datas[:, 0])
  gen_boards = get_gen_boards(decoder, model_results)
  adver_gen_boards = get_gen_boards(adver_decoder, model_results)

  save_metrics(eval_datas, gen_boards, adver_gen_boards, .95, non_train_indexies)

  with tf.io.gfile.GFile(os.path.join(FLAGS.job_dir, "eval_datas"), 'wb') as file:
    np.save(file, eval_datas)
  with tf.io.gfile.GFile(os.path.join(FLAGS.job_dir, "gen_boards"), 'wb') as file:
    np.save(file, gen_boards)
  with tf.io.gfile.GFile(os.path.join(FLAGS.job_dir, "adver_gen_boards"), 'wb') as file:
    np.save(file, adver_gen_boards)

if __name__ == '__main__':
  app.run(main)

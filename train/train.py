import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import os

import train.visualize_metric
from train.data_functions import num_black_cells, gen_data_batch, get_batch
from train.model_functions import create_models

FLAGS = flags.FLAGS
flags.DEFINE_integer('eval_data_size', 10000, '')
flags.DEFINE_integer('eval_interval', 1000, '')
flags.DEFINE_integer('max_train_steps', 80000, '')
flags.DEFINE_integer('count_cells', 0, '')
flags.DEFINE_integer('use_autoencoder', 1, '')

flags.DEFINE_float('target_task_metric_val', 2, '')
flags.DEFINE_float('target_pred_state_metric_val', .01, '')

flags.DEFINE_integer('use_adverse', 1, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('learning_rate', .001, '')
flags.DEFINE_float('reg_amount', 0.0, '')

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')


def get_train_model(model, datas, targets, decoder, decoder_task, discriminator, task_loss_fn,
                    train_index, indexes_to_adver, should_train_model, metric_prefix, task_metric, target_task_metric_val):
  """ This training function was designed to be flexible to work with a variety of tasks.
  The targets, decoder_task, task_loss_fn, task_metric and target_task_metric_val params should be different depending on the task.

  @param model:
  @param datas: The game of life board states
  @param targets: The data for the network to predict.
  @param decoder: The decoder to use to predict the game of life state.
    This is used for adversarial or autoencoder training even when training on a different task.
  @param decoder_task: The decoder to use for the task. Same as decoder if predicting life states.
  @param discriminator:
  @param task_loss_fn: The loss function to use for the task.
  @param train_index: The index to calculate the loss of the model on the task.
    If it's -1, it won't do task training, only adversarial or autoencoder.
  @param indexes_to_adver:
  @param should_train_model: If false, it only trains the decoder.
  @param metric_prefix:
  @param task_metric: A metric that outputs a value greater than 0. Lower is better.
  @param target_task_metric_val: The training stops once the metric is lower than this value.
  @return: A function to use for training
  """

  discrim_acc_metric = tf.keras.metrics.BinaryAccuracy()
  gen_acc_metric = tf.keras.metrics.BinaryAccuracy()
  reg_loss_metric = tf.keras.metrics.Mean()
  metrics = [
    ["task_metric", task_metric],
    ["discrim_acc", discrim_acc_metric],
    ["gen_acc", gen_acc_metric],
    ["reg loss", reg_loss_metric],
  ]
  acc_metrics = [tf.keras.metrics.BinaryAccuracy() for _ in range(FLAGS.num_timesteps+1)]
  for i in range(FLAGS.num_timesteps+1):
    metrics.append(["acc at {}".format(i), acc_metrics[i]])

  optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  discriminator_opt = tf.keras.optimizers.Adam()

  def calc_discriminator_loss(discrim_on_real, discrim_on_gen):
    real_loss = loss_fn(tf.ones_like(discrim_on_real), discrim_on_real)
    discrim_acc_metric.update_state(tf.ones_like(discrim_on_real), discrim_on_real)

    fake_loss = loss_fn(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    discrim_acc_metric.update_state(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    total_loss = real_loss + fake_loss
    return total_loss

  @tf.function
  def train_step(batch, batch_targets, adver_batch):
    inputs_batch = batch[:, 0]
    outputs_batch = batch
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
      generator_loss = 0.0
      discriminator_loss = 0.0
      ran_discrim = False
      model_outputs = model(inputs_batch)

      reg_loss = sum(model.losses) + sum(decoder.losses)
      reg_loss_metric.update_state([reg_loss])
      loss = reg_loss * FLAGS.reg_amount

      for i in range(FLAGS.num_timesteps+1):
        pred = decoder(model_outputs[i])
        if i == 0 and FLAGS.use_autoencoder:
          loss += loss_fn(outputs_batch[:, i], pred)
          if train_index == -1:
            task_metric.update_state(outputs_batch[:, i], pred)
        if i == train_index:
          task_pred = decoder_task(model_outputs[i])
          loss += task_loss_fn(batch_targets, task_pred)
          task_metric.update_state(batch_targets, task_pred)

        acc_metrics[i].update_state(outputs_batch[:, i], pred)

        if i in indexes_to_adver:
          discrim_on_pred = discriminator(tf.math.sigmoid(pred))
          discrim_on_real = discriminator(adver_batch[:,0])
          discriminator_loss += calc_discriminator_loss(discrim_on_real, discrim_on_pred)
          generator_loss = loss_fn(tf.ones_like(discrim_on_pred), discrim_on_pred)
          gen_acc_metric.update_state(tf.ones_like(discrim_on_pred), discrim_on_pred)
          loss += generator_loss
          ran_discrim = True

      loss /= (1+FLAGS.use_autoencoder)

    trainable_weights = []
    trainable_weights += decoder.trainable_weights
    if decoder != decoder_task:
      trainable_weights += decoder_task.trainable_weights
    if should_train_model:
      trainable_weights += model.trainable_weights

    grads = tape.gradient(loss, trainable_weights)
    clip_val = .1
    grads = [(tf.clip_by_value(grad, -clip_val, clip_val)) for grad in grads]
    optimizer.apply_gradients(zip(grads, trainable_weights))

    if ran_discrim:
      disctim_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
      discriminator_opt.apply_gradients(zip(disctim_grads, discriminator.trainable_weights))
    
  def train_full():
    writer = tf.summary.create_file_writer(FLAGS.job_dir)
    eval_datas = gen_data_batch(int(FLAGS.eval_data_size / 100) + 1, FLAGS.num_timesteps)
    non_train_indexies = range(1, FLAGS.num_timesteps)

    last_train_metric = acc_metrics[train_index]

    for name, metric in metrics:
      metric.reset_states()

    for step_i in range(FLAGS.max_train_steps):
      batch, batch_targets = get_batch(datas, targets, FLAGS.batch_size)
      adver_batch, _ = get_batch(datas, targets, FLAGS.batch_size)

      train_step(tf.constant(batch), tf.constant(batch_targets), tf.constant(adver_batch))
      if step_i % FLAGS.eval_interval == 0:
        with writer.as_default():
          for name, metric in metrics:
            print(step_i, name, metric.result().numpy())
            tf.summary.scalar(metric_prefix + "/" +name, metric.result(), step=step_i)

          model_results = model(eval_datas[:, 0])
          gen_boards = get_gen_boards(decoder, model_results)
          visualize_metric_result = train.visualize_metric.visualize_metric(eval_datas, gen_boards, .95, non_train_indexies)
          print("visualize_metric_result", visualize_metric_result)
          tf.summary.scalar(metric_prefix + "/" + "visualize_metric_result", visualize_metric_result, step=step_i)

        writer.flush()

        print("=" * 100, flush=True)
        if task_metric.result().numpy() < target_task_metric_val and step_i > 0:
          return

        for name, metric in metrics:
          metric.reset_states()

  return train_full

def np_sig(x):
  return 1/(1 + np.exp(-x)) 


def get_gen_boards(decoder, model_results):
  gen_boards = []
  for i in range(len(model_results)):
    gen_boards.append(decoder(model_results[i]).numpy())

  gen_boards = np.array(gen_boards)
  gen_boards = np_sig(gen_boards)
  gen_boards = np.transpose(gen_boards, (1,0,2,3,4))
  return gen_boards

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0)
mse_loss = tf.keras.losses.MeanSquaredError()

def save_metric_result(metric_result, metric_name):
  writer = tf.summary.create_file_writer(FLAGS.job_dir)
  with writer.as_default():
    tf.summary.scalar(metric_name, metric_result, step=0)
  writer.flush()

def save_metrics(eval_datas, gen_boards, adver_gen_boards, thresh, non_train_indexies):
  adver_metric = train.visualize_metric.visualize_metric(eval_datas, adver_gen_boards, thresh, non_train_indexies)
  regular_metric = train.visualize_metric.visualize_metric(eval_datas, gen_boards, thresh, non_train_indexies)
  if FLAGS.use_adverse:
    save_metric_result(adver_metric, "final_metric_result")
  else:
    save_metric_result(regular_metric, "final_metric_result")

class BinaryAccuracyInverseMetric(tf.keras.metrics.BinaryAccuracy):
  def result(self):
    return 1 - super().result()

def main(_):
  datas = gen_data_batch(100000, FLAGS.num_timesteps)
  eval_datas = gen_data_batch(FLAGS.eval_data_size, FLAGS.num_timesteps)
  encoder, intermediates, decoder, adver_decoder, decoder_counter, model, discriminator = create_models()

  pred_state_metric = BinaryAccuracyInverseMetric()
  non_train_indexies = range(1, FLAGS.num_timesteps)
  # Change the inputs to the train function depending on count_cells.
  if FLAGS.count_cells:
    task_metric = tf.keras.metrics.MeanSquaredError()
    targets = num_black_cells(datas[:, FLAGS.num_timesteps])
    task_loss_fn = mse_loss
    decoder_task = decoder_counter
    target_metric_val = FLAGS.target_task_metric_val
  else:
    task_metric = pred_state_metric
    task_loss_fn = loss_fn
    targets = datas[:, FLAGS.num_timesteps]
    decoder_task = decoder
    target_metric_val = FLAGS.target_pred_state_metric_val

  print("Full model training")
  get_train_model(model, datas, targets, decoder, decoder_task, discriminator, task_loss_fn,
                      FLAGS.num_timesteps, [], True, "train_full_model", task_metric, target_metric_val)()

  if task_metric.result().numpy() > FLAGS.target_task_metric_val:
    save_metric_result(-task_metric.result().numpy(), "final_metric_result")
    return

  if FLAGS.count_cells:
    train_index = -1  # This makes it only train autoencoder and adversarial, not task loss.
  else:
    train_index = FLAGS.num_timesteps

  # The decoder only and adverarial training uses the pred_state metric cause it's not doing any task specific training.
  print("Training Only Decoder", flush=True)
  get_train_model(model, datas, targets, adver_decoder, adver_decoder, discriminator, task_loss_fn,
                      train_index, [], False, "train_decoder", pred_state_metric, FLAGS.target_pred_state_metric_val+.05)()

  print("Training Only Decoder Adversarial")
  get_train_model(model, datas, targets, adver_decoder, adver_decoder, discriminator, task_loss_fn,
                      train_index, non_train_indexies, False, "train_decoder_adversarial", pred_state_metric, FLAGS.target_pred_state_metric_val+.01)()

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

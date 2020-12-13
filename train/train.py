import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import os

import train.visualize_metric
from train.data_functions import num_black_cells, gen_data_batch, get_batch
from train.model_functions import create_models
from collections import namedtuple


FLAGS = flags.FLAGS
flags.DEFINE_integer('eval_data_size', 10000, '')
flags.DEFINE_integer('eval_interval', 1000, '')
flags.DEFINE_integer('max_train_steps', 80000, '')
flags.DEFINE_integer('count_cells', 0, '')
flags.DEFINE_integer('use_autoencoder', 1, '')
flags.DEFINE_integer('use_task_autoencoder', 1, '')

flags.DEFINE_float('target_task_metric_val', .01, '')
flags.DEFINE_float('target_pred_state_metric_val', .01, '')

flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('learning_rate', .001, '')
flags.DEFINE_float('lr_decay_rate_per1M_steps', .9, '')
flags.DEFINE_float('reg_amount', 0.0, '')

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')

# class TaskInfo:
#   def __init__(self, name, train_indexes, data_fn, decoder, loss_fn, metric_class, target_metric_val):
#     self.name, train_indexes, data_fn, decoder, loss_fn, metric_class, target_metric_val

TaskInfo = namedtuple("TaskInfo", ['name', 'train_indexes', 'data_fn', 'decoder', 'loss_fn', 'metric_class', 'target_metric_val'])

def get_train_model(task_infos, model, datas, discriminator, indexes_to_adver, non_train_indexies, should_train_model,
                    adversarial_task_name, metric_stop_task_name, metric_prefix):

  discrim_acc_metric = tf.keras.metrics.BinaryAccuracy()
  gen_acc_metric = tf.keras.metrics.BinaryAccuracy()
  reg_loss_metric = tf.keras.metrics.Mean()
  metrics = [
    ["discrim_acc", discrim_acc_metric],
    ["gen_acc", gen_acc_metric],
    ["reg loss", reg_loss_metric],
  ]

  trainable_weights = []
  if should_train_model:
    trainable_weights += model.trainable_weights

  total_train_timesteps = len(indexes_to_adver)
  for task_info in task_infos:
    task_info['metrics'] = [task_info['metric_class']() for _ in range(FLAGS.num_timesteps+1)]
    for i in range(FLAGS.num_timesteps + 1):
      metrics.append(["{}_metric_at_{}".format(task_info['name'], i), task_info['metrics'][i]])

    total_train_timesteps += len(task_info['train_indexes'])

    trainable_weights.append(task_info['decoder'].trainable_weights)

    if "metric_stop_task_name" == task_info["name"]:
      metric_stop_task = task_info

  print("metrics", metrics)

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=FLAGS.learning_rate,
                                                               decay_steps=100000,
                                                               decay_rate=FLAGS.lr_decay_rate_per1M_steps)
  optimizer = tf.keras.optimizers.Adam(lr_schedule)
  discriminator_opt = tf.keras.optimizers.Adam()

  def calc_discriminator_loss(discrim_on_real, discrim_on_gen):
    real_loss = loss_fn(tf.ones_like(discrim_on_real), discrim_on_real)
    discrim_acc_metric.update_state(tf.ones_like(discrim_on_real), discrim_on_real)

    fake_loss = loss_fn(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    discrim_acc_metric.update_state(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    total_loss = real_loss + fake_loss
    return total_loss

  @tf.function
  def train_step(batch, adver_batch):
    inputs_batch = batch[:, 0]
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
      generator_loss = 0.0
      discriminator_loss = 0.0
      ran_discrim = False
      model_outputs = model(inputs_batch)
      loss = 0

      # TODO: Implement regularization
      for i in range(FLAGS.num_timesteps+1):
        for task_info in task_infos:
          batch_targets = task_info['data_fn'](batch)

          pred = task_info['decoder'](model_outputs[i])
          task_info['metrics'][i].update_state(batch_targets[:, i], pred)
          if i in task_info['train_indexes']:
            loss += loss_fn(batch_targets[:, i], pred)

          if (task_info["name"] == adversarial_task_name) and (i in indexes_to_adver):
            discrim_on_pred = discriminator(tf.math.sigmoid(pred))
            discrim_on_real = discriminator(adver_batch[:,0])
            discriminator_loss += calc_discriminator_loss(discrim_on_real, discrim_on_pred)
            generator_loss = loss_fn(tf.ones_like(discrim_on_pred), discrim_on_pred)
            gen_acc_metric.update_state(tf.ones_like(discrim_on_pred), discrim_on_pred)
            loss += generator_loss
            ran_discrim = True

      loss /= total_train_timesteps

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

    for name, metric in metrics:
      metric.reset_states()

    for step_i in range(FLAGS.max_train_steps):
      batch, batch_targets = get_batch(datas, FLAGS.batch_size)
      adver_batch, _ = get_batch(datas, FLAGS.batch_size)

      train_step(tf.constant(batch), tf.constant(batch_targets), tf.constant(adver_batch))
      if step_i % FLAGS.eval_interval == 0:
        with writer.as_default():
          for name, metric in metrics:
            print(step_i, name, metric.result().numpy())
            tf.summary.scalar(metric_prefix + "/" +name, metric.result(), step=step_i)

        # TODO: Implement visualize metric

        writer.flush()

        print("=" * 100, flush=True)
        metric_index = metric_stop_task['train_indexes'][-1]
        if metric_stop_task['metrics'][metric_index].result().numpy() < metric_stop_task['target_metric_val'] and step_i > 0:
          return

        for name, metric in metrics:
          metric.reset_states()

  return train_full

def np_sig(x):
  return 1/(1 + np.exp(-x)) 


def get_gens(decoder, model_results, is_img):
  gen_boards = []
  for i in range(len(model_results)):
    gen_boards.append(decoder(model_results[i]).numpy())
  gen_boards = np.array(gen_boards)

  if is_img:
    gen_boards = np_sig(gen_boards)
    gen_boards = np.transpose(gen_boards, (1,0,2,3,4))
  else:
    gen_boards = np.transpose(gen_boards, (1, 0, 2))
    gen_boards = np.squeeze(gen_boards, 2)
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
  save_metric_result(adver_metric, "adver_metric_result")
  save_metric_result(regular_metric, "regular_metric_result")
  save_metric_result(max(regular_metric, adver_metric), "final_metric_result")

class BinaryAccuracyInverseMetric(tf.keras.metrics.BinaryAccuracy):
  def result(self):
    return 1 - super().result()

def save_np(data, name):
  with tf.io.gfile.GFile(os.path.join(FLAGS.job_dir, name), 'wb') as file:
    np.save(file, data)

def main(_):
  datas = gen_data_batch(100000, FLAGS.num_timesteps)
  eval_datas = gen_data_batch(FLAGS.eval_data_size, FLAGS.num_timesteps)
  encoder, intermediates, decoder, adver_decoder, decoder_counter, model, discriminator = create_models()

  pred_state_metric = BinaryAccuracyInverseMetric()

  # Change the inputs to the train function depending on count_cells.
  if FLAGS.count_cells:
    non_train_indexies = range(1, FLAGS.num_timesteps+1) # Also use the last ts for adver training and metric.
    # Because the last timestep isn't trained to represent a game of life state.
    print("non_train_indexies", non_train_indexies)
    task_metric = tf.keras.metrics.MeanSquaredError()
    targets = num_black_cells(datas)
    task_loss_fn = mse_loss
    decoder_task = decoder_counter
    target_metric_val = FLAGS.target_task_metric_val
  else:
    non_train_indexies = range(1, FLAGS.num_timesteps)
    task_metric = pred_state_metric
    task_loss_fn = loss_fn
    targets = datas
    decoder_task = decoder
    target_metric_val = FLAGS.target_pred_state_metric_val

  # TaskInfo = namedtuple("TaskInfo",
  #                       ['name', 'train_indexes', 'data_fn', 'decoder', 'loss_fn', 'metric_class', 'target_metric_val'])
  #

  task_infos = [
    {'name': 'board', 'train_indexes': [0], 'data_fn': lambda x: x, 'decoder': decoder,
      'loss_fn': loss_fn, 'metric_class': BinaryAccuracyInverseMetric, 'target_metric_val': FLAGS.target_pred_state_metric_val},
    {'name': 'count', 'train_indexes': [0, FLAGS.num_timesteps - 1], 'data_fn': num_black_cells, 'decoder': decoder_counter,
     'loss_fn': mse_loss, 'metric_class': tf.keras.metrics.MeanSquaredError, 'target_metric_val': FLAGS.target_pred_state_metric_val}
  ]

  print("Full model training")
  get_train_model(task_infos, model, datas, discriminator, [], non_train_indexies, True, None, 'board', 'full_model')()

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
                      train_index, [], non_train_indexies, False, "train_decoder", pred_state_metric, FLAGS.target_pred_state_metric_val+.05)()

  print("Training Only Decoder Adversarial")
  get_train_model(model, datas, targets, adver_decoder, adver_decoder, discriminator, task_loss_fn,
                      train_index, non_train_indexies, non_train_indexies, False, "train_decoder_adversarial", pred_state_metric, FLAGS.target_pred_state_metric_val+.01)()

  model_results = model(eval_datas[:, 0])
  gen_boards = get_gens(decoder, model_results, True)
  adver_gen_boards = get_gens(adver_decoder, model_results, True)

  save_metrics(eval_datas, gen_boards, adver_gen_boards, .95, non_train_indexies)

  save_np(eval_datas, "eval_datas")
  save_np(gen_boards, "gen_boards")
  save_np(adver_gen_boards, "adver_gen_boards")

  if FLAGS.count_cells:
    task_gen = get_gens(decoder_task, model_results, False)
    save_np(task_gen, "task_gen")

if __name__ == '__main__':
  app.run(main)

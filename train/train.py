import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
import os

import train.visualize_metric
from train.data_functions import num_black_cells, gen_data_batch, get_batch
from train.model_functions import create_models, get_stop_grad_dec


FLAGS = flags.FLAGS
flags.DEFINE_integer('eval_data_size', 10000, '')
flags.DEFINE_integer('eval_interval', 5000, '')
flags.DEFINE_integer('max_train_steps', 80000, '')
flags.DEFINE_integer('count_cells', 0, '')
flags.DEFINE_integer('use_autoencoder', 1, '')
flags.DEFINE_integer('use_task_autoencoder', 1, '')

flags.DEFINE_float('target_task_metric_val', 0.001, '')
flags.DEFINE_float('target_pred_state_metric_val', .01, '')
flags.DEFINE_float('early_task_metric_val', 1.0, '')
flags.DEFINE_float('early_pred_state_metric_val', 1.0, '')
flags.DEFINE_integer('early_stop_step', 100000, '')

flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('training_data_amount', 200000, '')
flags.DEFINE_integer('start_curiculum_board_size', 4, '')
flags.DEFINE_float('learning_rate', .001, '')
flags.DEFINE_float('lr_decay_rate_per1M_steps', .9, '')
flags.DEFINE_float('reg_amount', 0.0, '')

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')

def get_train_model(task_infos, model, datas, discriminator, should_train_model,
                    adversarial_task_name, metric_stop_task_name, metric_prefix, start_curiculum_board_size, max_train_steps=None):
  """This training function was designed to be flexible to work with a variety of tasks.

  @ param task_infos: A list of dicts. Each dict specifies the parameters of a task. Keys:
    name, train_indexes: The indexes train the task on,
    data_fn: A function called on each batch to convert it to the format to train the task,
    decoder, loss_fn: Loss to use between the decoder predictions and data,
    metric_class: Metrics for each timepstep are created using this class,
    target_metric_val: Training will stop when the metric gets below this value,
    early_metric_val: Training will stop if the metric is above this value after x steps
  @param model:
  @param datas: The game of life board states
  @param discriminator:
  @param should_train_model: If false, it only trains the decoder.
  @param adversarial_task_name: The task with the decoder to train adversarially.
  @param metric_stop_task_name: The task with the metric to decide when to stop training.
  @param metric_prefix:
  @return: A function to use for training
  """

  if not max_train_steps:
    max_train_steps = FLAGS.max_train_steps

  discrim_acc_metric = tf.keras.metrics.BinaryAccuracy()
  gen_acc_metric = tf.keras.metrics.BinaryAccuracy()
  reg_loss_metric = tf.keras.metrics.Mean()
  metrics = [
    ["discrim_acc", discrim_acc_metric],
    ["gen_acc", gen_acc_metric],
    ["reg loss", reg_loss_metric],
  ]
  for task_info in task_infos:
    task_info['metrics'] = [task_info['metric_class']() for _ in range(FLAGS.num_timesteps+1)]
    for i in range(FLAGS.num_timesteps + 1):
      metrics.append(["{}_metric_at_{}".format(task_info['name'], i), task_info['metrics'][i]])

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
      discriminator_loss = 0.0
      ran_discrim = False
      model_outputs = model(inputs_batch)
      loss = 0

      for i in range(FLAGS.num_timesteps+1):
        for task_info in task_infos:
          batch_targets = task_info['data_fn'](batch)[:, i]
          pred = task_info['decoder'](model_outputs[i])

          task_info['metrics'][i].update_state(batch_targets, pred)
          if i in task_info['train_indexes']:
            loss += task_info['loss_fn'](batch_targets, pred)

          elif task_info["name"] == adversarial_task_name:
            discrim_on_pred = discriminator(pred)
            discrim_on_real = discriminator(adver_batch[:,0])
            discriminator_loss += calc_discriminator_loss(discrim_on_real, discrim_on_pred)
            generator_loss = loss_fn(tf.ones_like(discrim_on_pred), discrim_on_pred)
            gen_acc_metric.update_state(tf.ones_like(discrim_on_pred), discrim_on_pred)
            loss += generator_loss
            ran_discrim = True

      reg_loss = sum(model.losses)
      for task_info in task_infos:
        reg_loss += sum(task_info['decoder'].losses)
      reg_loss_metric.update_state([reg_loss])
      loss += reg_loss * FLAGS.reg_amount

    trainable_weights = []
    if should_train_model:
      trainable_weights += model.trainable_weights
    for task_info in task_infos:
      trainable_weights += task_info['decoder'].trainable_weights

    grads = tape.gradient(loss, trainable_weights)
    grads_weights = list(zip(grads, trainable_weights))
    clip_val = .1
    grads_weights = [grad_weight for grad_weight in grads_weights if grad_weight[0] != None]
    grads_weights = [(tf.clip_by_value(grad, -clip_val, clip_val), weight) for grad, weight in grads_weights]
    optimizer.apply_gradients(grads_weights)

    if ran_discrim:
      disctim_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
      discriminator_opt.apply_gradients(zip(disctim_grads, discriminator.trainable_weights))
    
  def train_full():
    board_size = start_curiculum_board_size
    writer = tf.summary.create_file_writer(FLAGS.job_dir)

    for name, metric in metrics:
      metric.reset_states()

    for step_i in range(max_train_steps):
      batch = get_batch(datas[board_size], FLAGS.batch_size)
      adver_batch = get_batch(datas[board_size], FLAGS.batch_size)

      train_step(tf.constant(batch), tf.constant(adver_batch))

      if step_i == FLAGS.early_stop_step and board_size == start_curiculum_board_size:
        task_good_enough, _ = is_task_good_enough(task_infos, metric_stop_task_name, 'early_metric_val')
        if not task_good_enough:
          print("STOP not promising enough", flush=True)
          return

      if step_i % FLAGS.eval_interval == 0:
        with writer.as_default():
          for name, metric in metrics:
            tf.summary.scalar(metric_prefix + "/" +name, metric.result(), step=step_i)
            print(metric_prefix + "/" + name, metric.result())

          tf.summary.scalar(metric_prefix + "/" + "board_size", board_size, step=step_i)
          print("board_size", board_size)
        writer.flush()

        task_good_enough, _ = is_task_good_enough(task_infos, metric_stop_task_name, 'target_metric_val')
        if task_good_enough and step_i > 0:
          if board_size >= FLAGS.board_size:
            return
          else:
            board_size += 1
        if step_i >= max_train_steps - 3:
          return  # So the metric values don't get reset when there's only a few timesteps left.

        for name, metric in metrics:
          metric.reset_states()
  return train_full

def is_task_good_enough(task_infos, metric_stop_task_name, target_val_name):
  for task_info in task_infos:
    if metric_stop_task_name == task_info["name"]:
      metric_stop_task = task_info

  metric_index = max(metric_stop_task['train_indexes'])
  stop_metric = metric_stop_task['metrics'][metric_index]

  metric_result = stop_metric.result().numpy()
  print("stop metric result", metric_result, target_val_name, metric_stop_task[target_val_name])
  return metric_result < metric_stop_task[target_val_name], metric_result

def get_gens(decoder, model_results, is_img):
  gen_boards = []
  for i in range(len(model_results)):
    gen_boards.append(decoder(model_results[i]).numpy())
  gen_boards = np.array(gen_boards)

  if is_img:
    gen_boards = np.clip(gen_boards, 0.0, 1.0)
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

class CountAccuracyInverseMetric(tf.keras.metrics.Accuracy):
  """Gives 1- the accuracy whe the prediction is rounded to the nearest integer."""
  def convert_y(self, y):
    return tf.math.round(y * (FLAGS.board_size ** 2))

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = self.convert_y(y_true)
    y_pred = self.convert_y(y_pred)

    return super().update_state(y_true, y_pred, sample_weight)

  def result(self):
    return 1 - super().result()


def save_np(data, name):
  with tf.io.gfile.GFile(os.path.join(FLAGS.job_dir, name), 'wb') as file:
    np.save(file, data)

def main(_):
  datas = {}
  for board_size in range(FLAGS.start_curiculum_board_size, FLAGS.board_size+1):
    datas[board_size] = gen_data_batch(FLAGS.training_data_amount, FLAGS.num_timesteps, board_size)
    print("Gen data for", board_size)

  eval_datas = gen_data_batch(FLAGS.eval_data_size, FLAGS.num_timesteps, FLAGS.board_size)
  encoder, intermediates, decoder, adver_decoder, decoder_counter, model, discriminator = create_models()

  board_train_indexes = set()
  if FLAGS.use_autoencoder:
    board_train_indexes.add(0)
  if not FLAGS.count_cells:
    board_train_indexes.add(FLAGS.num_timesteps)

  count_train_indexes = set()
  if FLAGS.use_task_autoencoder:
    count_train_indexes.add(0)
  count_train_indexes.add(FLAGS.num_timesteps)

  if FLAGS.count_cells:
    metric_stop_task_name = 'count'
  else:
    metric_stop_task_name = 'board'

  task_infos = [
    {'name': 'board', 'train_indexes': board_train_indexes, 'data_fn': lambda x: x, 'decoder': decoder,
      'loss_fn': mse_loss, 'metric_class': BinaryAccuracyInverseMetric, 'target_metric_val': FLAGS.target_pred_state_metric_val,
     'early_metric_val': FLAGS.early_pred_state_metric_val}]
  if FLAGS.count_cells:
    task_infos.append(
      {'name': 'count', 'train_indexes': count_train_indexes, 'data_fn': num_black_cells, 'decoder': decoder_counter,
      'loss_fn': mse_loss, 'metric_class': CountAccuracyInverseMetric, 'target_metric_val': FLAGS.target_task_metric_val,
       'early_metric_val': FLAGS.early_task_metric_val})

  print("task_infos", task_infos, flush=True)
  print("Full model training")
  get_train_model(task_infos=task_infos, model=model, datas=datas, discriminator=None, should_train_model=True,
                  adversarial_task_name=None, metric_stop_task_name=metric_stop_task_name, metric_prefix='full_model',
                  start_curiculum_board_size=FLAGS.start_curiculum_board_size)()

  task_good_enough, task_metric_result = is_task_good_enough(task_infos, metric_stop_task_name, 'target_metric_val')
  if not task_good_enough:
    save_metric_result(-task_metric_result, "final_metric_result")
    return

  task_infos[0]['decoder'] = adver_decoder  # Use a different decoder
  task_infos = [task_infos[0]]  # Only train the board task for adversarial.
  print("task infos adversarial", task_infos)
  print("Training Only Decoder", flush=True)
  get_train_model(task_infos=task_infos, model=model, datas=datas, discriminator=discriminator, should_train_model=False,
                  adversarial_task_name=None, metric_stop_task_name='board', metric_prefix='train_decoder',
                  start_curiculum_board_size=FLAGS.board_size)()

  print("Training Only Decoder Adversarial")
  get_train_model(task_infos=task_infos, model=model, datas=datas, discriminator=discriminator, should_train_model=False,
                  adversarial_task_name='board', metric_stop_task_name='board', metric_prefix='train_decoder_adversarial',
                  start_curiculum_board_size=FLAGS.board_size)()

  model_results = model(eval_datas[:, 0])
  gen_boards = get_gens(decoder, model_results, True)
  adver_gen_boards = get_gens(adver_decoder, model_results, True)

  all_indexes = set(range(FLAGS.num_timesteps+1))
  # Only consider the indexes we train the board on as train indexes.
  # The indexes that count cells is trained on could still be non train indexes
  non_train_indexies = all_indexes - board_train_indexes
  save_metrics(eval_datas, gen_boards, adver_gen_boards, .95, non_train_indexies)

  save_np(eval_datas, "eval_datas")
  save_np(gen_boards, "gen_boards")
  save_np(adver_gen_boards, "adver_gen_boards")

  if FLAGS.count_cells:
    task_gen = get_gens(decoder_counter, model_results, False)
    save_np(task_gen, "task_gen")


  def fine_tune_new_decoder(train_indexes, name):
    print("Train decoder {}".format(name))
    task_infos[0]["train_indexes"] = train_indexes
    task_infos[0]["decoder"] = get_stop_grad_dec(2, "dec_{}".format(name), 4)
    print("task infos", task_infos)
    get_train_model(task_infos=task_infos, model=model, datas=datas, discriminator=None, should_train_model=False,
                      adversarial_task_name=None, metric_stop_task_name='board', metric_prefix='train_decoder_{}'.format(name),
                    max_train_steps=int(FLAGS.max_train_steps/10), start_curiculum_board_size=FLAGS.board_size)()
    new_dec_gen_boards = get_gens(task_infos[0]["decoder"], model_results, True)
    save_np(new_dec_gen_boards, "gen_boards_{}".format(name))

  fine_tune_new_decoder({0, FLAGS.num_timesteps}, "first_last")
  fine_tune_new_decoder(set(range(FLAGS.num_timesteps+1)), "all")
  for dec_ts in range(FLAGS.num_timesteps + 1):
    fine_tune_new_decoder({dec_ts}, dec_ts)

if __name__ == '__main__':
  app.run(main)

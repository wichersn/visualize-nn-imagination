import numpy as np
from scipy.signal import convolve2d
import tensorflow as tf
from absl import app
from absl import flags
import os

import train.visualize_metric

FLAGS = flags.FLAGS
flags.DEFINE_integer('eval_data_size', 10000, '')
flags.DEFINE_multi_integer('board_size', [20,20], '')
flags.DEFINE_integer('eval_interval', 1000, '')
flags.DEFINE_integer('max_train_steps', 80000, '')
flags.DEFINE_integer('count_cells', 0, '')

flags.DEFINE_integer('num_timesteps', 3, '')
flags.DEFINE_integer('encoded_size', 8, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('timestep_layers', 3, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('learning_rate', .001, '')
flags.DEFINE_float('reg_amount', 0.0, '')
flags.DEFINE_integer('use_residual', 1, '')
flags.DEFINE_integer('use_adverse', 1, '')

flags.DEFINE_string('job_dir', '',
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'job_dir')


#Training data functions
def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def num_black_cells(X):
    return np.sum(X, axis=(1, 2))

def convert_model_in(data):
  data = np.array(data)
  data = data.astype(np.float32)
  data = np.expand_dims(data, -1)
  return data

def gen_data_batch(size, skip):
  datas = []
  start_next = None
  for _ in range(size):
    if np.random.rand(1) < .1 or start_next is None:
      life_state = np.random.rand(FLAGS.board_size[0], FLAGS.board_size[1]) > .5
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

# Model and training.
def create_models(encoded_size, T, use_residual):
  input_shape = FLAGS.board_size + [1, ]
  input_layer = tf.keras.Input(shape=input_shape)

  encoder = tf.keras.Sequential(name="encoder")
  for _ in range(FLAGS.encoder_layers):
    encoder.add(tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),)
  print("encoder", encoder.layers)

  intermediates = [encoder(input_layer)]

  timestep_model = tf.keras.Sequential(name="timestep_model")
  for _ in range(FLAGS.timestep_layers):
   timestep_model.add(tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("timestep_model", timestep_model.layers)

  for i in range(T):
    timestep = timestep_model(intermediates[-1])
    if use_residual:
      timestep += intermediates[-1]
    intermediates.append(timestep)

  decoder = tf.keras.Sequential(name="decoder")
  for _ in range(FLAGS.decoder_layers-1):
    decoder.add(tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  decoder.add(tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("decoder", decoder.layers)

  decoder_counter = tf.keras.Sequential(name="decoder-counter")
  for _ in range(FLAGS.decoder_layers-1):
    decoder_counter.add(tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  decoder_counter.add(tf.keras.layers.Flatten())
  decoder_counter.add(tf.keras.layers.Dense(1))
  print("decoder_counter", decoder_counter.layers)

  model = tf.keras.Model(inputs=input_layer, outputs=intermediates)

  discriminator = tf.keras.Sequential(
      [
        tf.keras.layers.Conv2D(4, 3, strides=2, activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(8, 3, strides=2, activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(16, 3, strides=2, activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
      ], name="discriminator",
  )

  return encoder, intermediates, decoder, decoder_counter, model, discriminator

def calc_discriminator_loss(discrim_on_real, discrim_on_gen):
    real_loss = loss_fn(tf.ones_like(discrim_on_real), discrim_on_real)
    discrim_acc_metric.update_state(tf.ones_like(discrim_on_real), discrim_on_real)

    fake_loss = loss_fn(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    discrim_acc_metric.update_state(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    total_loss = real_loss + fake_loss
    return total_loss

def get_batch(datas, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    targets = num_black_cells(datas[:, 0])
    return datas[idx], targets[idx]

def get_train_model(model, discriminator, optimizer, datas, discriminator_opt, T, reg_amount=0):
  @tf.function
  def train_step(batch, batch_targets, adver_batch, decoder, decoder_counter, indexes_to_train, indexes_to_adver, train_model, metric_prefix):
    inputs_batch = batch[:, 0]
    outputs_batch = batch
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
      generator_loss = 0.0
      discriminator_loss = 0.0
      ran_discrim = False
      model_outputs = model(inputs_batch)

      reg_loss = sum(model.losses) + sum(decoder.losses)
      reg_loss_metric.update_state([reg_loss])
      loss = reg_loss * reg_amount

      if metric_prefix == "count_cells":
        pred = decoder_counter(model_outputs[-1])
        loss += mse_loss(batch_targets, pred)
        train_mse_metric_count_cells.update_state(batch_targets, pred)

      for i in range(T+1):
        pred = decoder(model_outputs[i])
        if i in indexes_to_train:
          loss += loss_fn(outputs_batch[:, i], pred)
        acc_metrics[i].update_state(outputs_batch[:, i], pred)

        if i in indexes_to_adver:
          discrim_on_pred = discriminator(tf.math.sigmoid(pred))
          discrim_on_real = discriminator(adver_batch[:,0])
          discriminator_loss += calc_discriminator_loss(discrim_on_real, discrim_on_pred)
          generator_loss = loss_fn(tf.ones_like(discrim_on_pred), discrim_on_pred)
          gen_acc_metric.update_state(tf.ones_like(discrim_on_pred), discrim_on_pred)
          loss += generator_loss
          ran_discrim = True

      loss /= len(indexes_to_train)
    trainable_weights = None
    if metric_prefix == "count_cells":
      trainable_weights = decoder_counter.trainable_weights
      trainable_weights += decoder.trainable_weights
    else:
      trainable_weights = decoder.trainable_weights
    if train_model:
      trainable_weights += model.trainable_weights
    grads = tape.gradient(loss, trainable_weights)
    clip_val = .1
    grads = [(tf.clip_by_value(grad, -clip_val, clip_val))
                                      for grad in grads]
    optimizer.apply_gradients(zip(grads, trainable_weights))

    if ran_discrim:
      disctim_grads = disc_tape.gradient(discriminator_loss, discriminator.trainable_weights)
      discriminator_opt.apply_gradients(zip(disctim_grads, discriminator.trainable_weights))
    
  def train_model(decoder, decoder_counter, indexes_to_train, indexes_to_adver, train_model, stop_acc, stop_mse, metric_prefix):
    writer = tf.summary.create_file_writer(FLAGS.job_dir)
    eval_datas = gen_data_batch(int(FLAGS.eval_data_size/100)+1, FLAGS.num_timesteps)
    non_train_indexies = range(1, FLAGS.num_timesteps)

    print("last train index", indexes_to_train[-1])
    last_train_metric = acc_metrics[indexes_to_train[-1]]

    for name, metric in metrics:
      metric.reset_states()

    for step_i in range(FLAGS.max_train_steps):
      batch, targets = get_batch(datas, FLAGS.batch_size)
      adver_batch, adver_targets = get_batch(datas, FLAGS.batch_size)

      train_step(tf.constant(batch), tf.constant(targets), tf.constant(adver_batch), decoder, decoder_counter, indexes_to_train, indexes_to_adver, train_model, metric_prefix)
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
        if metric_prefix == "count_cells" and train_mse_metric_count_cells.result().numpy() < stop_mse and step_i > 0:
          return train_mse_metric_count_cells.result().numpy()
        if metric_prefix != "count_cells" and last_train_metric.result().numpy() > stop_acc and step_i > 0:
          return last_train_metric.result().numpy()

        for name, metric in metrics:
          metric.reset_states()
    if metric_prefix == "count_cells":
      return train_mse_metric_count_cells.result().numpy()
    else:
      return last_train_metric.result().numpy()

  return train_model

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


train_mse_metric_count_cells = tf.keras.metrics.MeanSquaredError()
discrim_acc_metric = tf.keras.metrics.BinaryAccuracy()
gen_acc_metric = tf.keras.metrics.BinaryAccuracy()
reg_loss_metric = tf.keras.metrics.Mean()
metrics = [
    ["train_mse", train_mse_metric_count_cells],
    ["discrim_acc", discrim_acc_metric],
    ["gen_acc", gen_acc_metric],
    ["reg loss", reg_loss_metric],
]
acc_metrics = None

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0)
mse_loss = tf.keras.losses.MeanSquaredError()
leak_relu = tf.keras.layers.LeakyReLU

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

  if FLAGS.count_cells:
    train_indexies = 0
  else:
    train_indexies = [0,FLAGS.num_timesteps]
  non_train_indexies = range(1, FLAGS.num_timesteps)
  target_train_accuracy = .99
  target_train_mse = 0.5
  print("Full model training")
  train_acc = get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, reg_amount=FLAGS.reg_amount)(
    decoder, decoder_counter, train_indexies, [], True, target_train_accuracy, target_train_mse, "train_full_model")

  if train_acc < target_train_accuracy:
    save_metric_result(train_acc - 1, "final_metric_result")
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
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, reg_amount=FLAGS.reg_amount)(
    adver_decoder, decoder_counter, train_indexies, [], False, .95, target_train_mse, "train_decoder")

  print("Training Only Decoder Adversarial")
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, FLAGS.num_timesteps, reg_amount=FLAGS.reg_amount)(
    adver_decoder, decoder_counter, train_indexies, non_train_indexies, False, .98, target_train_mse, "train_decoder_adversarial")

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

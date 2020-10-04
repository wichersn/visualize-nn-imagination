import numpy as np
import sys
from scipy.signal import convolve2d
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
flags.DEFINE_multi_integer('board_size', [20,20], '')
flags.DEFINE_integer('eval_interval', 500, '')
flags.DEFINE_integer('max_train_steps', 20000, '')

#Training data functions
def life_step(X):
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='fill') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def convert_model_in(data):
  data = np.array(data)
  data = data.astype(int)
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

  encoder = tf.keras.Sequential(
      [
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
      ], name="encoder",
  )
  intermediates = [encoder(input_layer)]
  all_hidden = []

  timestep_model = tf.keras.Sequential(
      [
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1), input_shape=FLAGS.board_size+[encoded_size,]),
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
      ], name="timestep_model",
  )
  timestep_model_all_hidden_out = [layer.output for layer in timestep_model.layers]
  timestep_model_all_hidden_out[-1] += timestep_model.layers[0].input
  timestep_model_all_hidden = tf.keras.Model(timestep_model.inputs, timestep_model_all_hidden_out)
  for i in range(T):
    timestep_all_hidden = timestep_model_all_hidden(intermediates[-1])
    if use_residual:
      timestep_all_hidden[-1] += intermediates[-1]
    intermediates.append(timestep_all_hidden[-1])
    all_hidden.append(timestep_all_hidden)

  decoder = tf.keras.Sequential(
      [
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
        tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),
      ], name="decoder",
  )

  model = tf.keras.Model(inputs=input_layer, outputs=intermediates)
  all_hidden_model = tf.keras.Model(inputs=input_layer, outputs=all_hidden)

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

  return encoder, intermediates, all_hidden, decoder, model, all_hidden_model, discriminator

def calc_discriminator_loss(discrim_on_real, discrim_on_gen):
    real_loss = loss_fn(tf.ones_like(discrim_on_real), discrim_on_real)
    discrim_acc_metric.update_state(tf.ones_like(discrim_on_real), discrim_on_real)

    fake_loss = loss_fn(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    discrim_acc_metric.update_state(tf.zeros_like(discrim_on_gen), discrim_on_gen)
    total_loss = real_loss + fake_loss
    return total_loss

def get_batch(datas, batch_size):
    idx = np.random.choice(np.arange(len(datas)), batch_size, replace=False)
    return datas[idx]

def get_train_model(model, discriminator, optimizer, datas, discriminator_opt, T, reg_amount=0):
  @tf.function
  def train_step(batch, adver_batch, decoder, indexes_to_train, indexes_to_adver, train_model):
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

      for i in range(T+1):
        pred = decoder(model_outputs[i])
        if i in indexes_to_train:
          loss += loss_fn(outputs_batch[:, i], pred)
          train_acc_metric.update_state(outputs_batch[:, i], pred)
        else:
          non_train_acc_metric.update_state(outputs_batch[:, i], pred)

        if i in indexes_to_adver:
          discrim_on_pred = discriminator(tf.math.sigmoid(pred))
          discrim_on_real = discriminator(adver_batch[:,0])
          discriminator_loss += calc_discriminator_loss(discrim_on_real, discrim_on_pred)
          generator_loss = loss_fn(tf.ones_like(discrim_on_pred), discrim_on_pred)
          gen_acc_metric.update_state(tf.ones_like(discrim_on_pred), discrim_on_pred)
          loss += generator_loss
          ran_discrim = True

      loss /= len(indexes_to_train)

    trainable_weights =  decoder.trainable_weights
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
    
  def train_model(decoder, indexes_to_train, indexes_to_adver, train_model, stop_acc):
    for name, metric in metrics:
      metric.reset_states()

    for step_i in range(FLAGS.max_train_steps):
      batch = get_batch(datas, 128)
      adver_batch = get_batch(datas, 128)

      train_step(tf.constant(batch), tf.constant(adver_batch), decoder, indexes_to_train, indexes_to_adver, train_model)
      if step_i % FLAGS.eval_interval == 0:
        for name, metric in metrics:
          print(step_i, name, metric.result().numpy())
        print("=" * 100, flush=True)

        if train_acc_metric.result().numpy() > stop_acc and step_i > 0:
            break

        for name, metric in metrics:
          metric.reset_states()

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

def single_index_metric(gt, gen, thresh):
  """Returns the porportion of gen instances close enough to the ground truth."""
  equal = np.equal(gt, gen>.5)
  acc = np.mean(equal, (1,2,3))
  over_thresh = acc >= thresh
  return np.mean(over_thresh)

def single_gt_index_metric(gt, gen_boards, thresh, non_train_indexies):
  """Metric representing how well the gen boards represent the single ground truth state.
  
  It gets full credit the time it predicts the state the best, and partial credit for all other times.
  """
  metrics_for_gt = []
  for j in non_train_indexies:
    metrics_for_gt.append(single_index_metric(gt, gen_boards[:, j], thresh))
  metrics_for_gt.sort(reverse = True)
  weights = np.ones([len(non_train_indexies)]) * .5
  weights[0] = 1
  result = sum(np.multiply(metrics_for_gt, weights))
  return result

def metric(eval_datas, gen_boards, thresh, non_train_indexies):
  """Averages the metric on each of the ground truth states."""
  total_metric = 0
  for i in non_train_indexies:
    total_metric += single_gt_index_metric(eval_datas[:, i], gen_boards, thresh, non_train_indexies)
  return total_metric / float(len(non_train_indexies))

def combine_metric(eval_datas, gen_boards, adver_gen_boards, thresh, non_train_indexies):
  adver_metric = metric(eval_datas, adver_gen_boards, thresh, non_train_indexies)
  regular_metric = metric(eval_datas, gen_boards, thresh, non_train_indexies)
  return max(adver_metric, regular_metric)

train_acc_metric = tf.keras.metrics.BinaryAccuracy()
non_train_acc_metric = tf.keras.metrics.BinaryAccuracy()
discrim_acc_metric = tf.keras.metrics.BinaryAccuracy()
gen_acc_metric = tf.keras.metrics.BinaryAccuracy()
reg_loss_metric = tf.keras.metrics.Mean()
metrics = [
    ["train_acc", train_acc_metric],
    ["non_train_acc", non_train_acc_metric],
    ["discrim_acc", discrim_acc_metric],
    ["gen_acc", gen_acc_metric],
    ["reg loss", reg_loss_metric],
]

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0)
leak_relu = tf.keras.layers.LeakyReLU

def main(_):
  T = 3
  encoded_size = 32
  datas = gen_data_batch(100000, T)
  eval_datas = gen_data_batch(10000, T)
  encoder, intermediates, all_hidden, decoder, model, all_hidden_model, discriminator = create_models(encoded_size, T, False)

  optimizer=tf.keras.optimizers.Adam(.001) # .001 lr works better than smaller ones.
  # TODO: Lower learning rate as non train accuracy improves. Might help not mess up the hidden representations that it learned.
  discriminator_opt=tf.keras.optimizers.Adam()

  train_indexies = [0,T]
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, T, reg_amount=0)(decoder, train_indexies, [], True, .99)

  adver_decoder = tf.keras.Sequential(
      [
        tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x)),
        tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same'),
        tf.keras.layers.Conv2D(1, 3, activation=None, padding='same'),
      ], name="adver_decoder",
  )

  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, T, reg_amount=0)(adver_decoder, train_indexies, [], False, .99)
  print("Training adversarally")
  non_train_indexies = range(1, T)
  get_train_model(model, discriminator, optimizer, datas, discriminator_opt, T, reg_amount=0)(adver_decoder, train_indexies, non_train_indexies, False, .99)

  model_results = model(eval_datas[:, 0])
  gen_boards = get_gen_boards(decoder, model_results)
  adver_gen_boards = get_gen_boards(adver_decoder, model_results)

  combine_metric(eval_datas, gen_boards, adver_gen_boards, .95, non_train_indexies)


if __name__ == '__main__':
  app.run(main)
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('job_dir', '', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('model_save_dir', None, 'The dir to read saved checkpoints from. Defaults to job_dir')
flags.DEFINE_integer('model_timesteps', 4, '')
flags.DEFINE_integer('encoded_size', 8, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('timestep_layers', 3, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_integer('adver_decoder_layers', 2, '')
flags.DEFINE_integer('decoder_counter_layers', 2, 'Only used for the count cells task.')
flags.DEFINE_integer('decoder_counter_strides', 2, 'Only used for the count cells task.')
flags.DEFINE_integer('use_residual', 1, '')
flags.DEFINE_integer('use_rnn', 1, '')
flags.DEFINE_float('dropout_rate', 0.0, '')

leak_relu = tf.keras.layers.LeakyReLU

def create_timestep_model(name=''):
  timestep_model = tf.keras.Sequential(name="timestep_model"+name)
  for _ in range(FLAGS.timestep_layers):
    timestep_model.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(1)))
    timestep_model.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))
  print("timestep_model", timestep_model.layers)
  return timestep_model

def add_decoder_layers(decoder, decoder_layers, encoded_size=None):
  if not encoded_size:
    encoded_size = FLAGS.encoded_size
  for _ in range(decoder_layers-1):
    decoder.add(tf.keras.layers.Conv2D(encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
    decoder.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))

def get_stop_grad_dec(decoder_layers, name, encoded_size=None):
  decoder = tf.keras.Sequential(
    [
      tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x)),
    ], name=name
  )
  add_decoder_layers(decoder, decoder_layers, encoded_size)
  decoder.add(
    tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  print(name, decoder.layers)
  return decoder

def get_save_path(name):
  save_dir = FLAGS.model_save_dir
  if save_dir == None:
    save_dir = FLAGS.job_dir
  return os.path.join(save_dir, name+"_ckpt", 'ckpt')

def save_model(model, name):
  model.save_weights(get_save_path(name))

def maybe_load_model(model, name):
  try:
    model.load_weights(get_save_path(name))
    print(name + " model loaded")
  except tf.errors.NotFoundError:
    pass

def create_count_decoder():
  decoder_counter = tf.keras.Sequential(name="decoder-counter")
  add_decoder_layers(decoder_counter, FLAGS.decoder_counter_layers-1)
  decoder_counter.add(tf.keras.layers.Flatten())
  decoder_counter.add(tf.keras.layers.Dense(1))
  maybe_load_model(decoder_counter, 'decoder_counter')
  return decoder_counter

def create_decoder(name):
  decoder = tf.keras.Sequential(name=name)
  add_decoder_layers(decoder, FLAGS.decoder_layers)
  decoder.add(tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  maybe_load_model(decoder, name)
  return decoder

def create_gol_decoder():
  return create_decoder("decoder")

def create_patch_decoder():
  return create_decoder("patch_decoder")

def create_models():
  input_shape = [FLAGS.board_size, FLAGS.board_size] + [1, ]
  input_layer = tf.keras.Input(shape=input_shape)

  encoder = tf.keras.Sequential(name="encoder")
  for _ in range(FLAGS.encoder_layers):
    encoder.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),)
    encoder.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))
  print("encoder", encoder.layers)

  intermediates = [encoder(input_layer)]

  if FLAGS.use_rnn:
    timestep_model = create_timestep_model()
  for i in range(FLAGS.model_timesteps):
    if not FLAGS.use_rnn:
      timestep_model = create_timestep_model('_'+str(i))

    timestep = timestep_model(intermediates[-1])
    if FLAGS.use_residual:
      timestep += intermediates[-1]
    intermediates.append(timestep)

  model = tf.keras.Model(inputs=input_layer, outputs=intermediates)

  discriminator = tf.keras.Sequential(
      [
        tf.keras.layers.Conv2D(4, 3, strides=2, padding='same', activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(8, 3, strides=2, padding='same', activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(16, 3, strides=2, padding='same', activation=leak_relu()),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1),
      ], name="discriminator",
  )

  adver_decoder = get_stop_grad_dec(FLAGS.adver_decoder_layers, "adver_decoder")
  maybe_load_model(encoder, 'encoder')
  maybe_load_model(model, 'model')

  return encoder, intermediates, adver_decoder, model, discriminator

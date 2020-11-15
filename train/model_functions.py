import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_timesteps', 3, '')
flags.DEFINE_integer('encoded_size', 8, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('timestep_layers', 3, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_integer('use_residual', 1, '')
flags.DEFINE_integer('use_rnn', 1, '')

leak_relu = tf.keras.layers.LeakyReLU

def create_timestep_model(name=''):
  timestep_model = tf.keras.Sequential(name="timestep_model"+name)
  for _ in range(FLAGS.timestep_layers):
   timestep_model.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("timestep_model", timestep_model.layers)
  return timestep_model

def create_models():
  input_shape = FLAGS.board_size + [1, ]
  input_layer = tf.keras.Input(shape=input_shape)

  encoder = tf.keras.Sequential(name="encoder")
  for _ in range(FLAGS.encoder_layers):
    encoder.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)),)
  print("encoder", encoder.layers)

  intermediates = [encoder(input_layer)]

  if FLAGS.use_rnn:
    timestep_model = create_timestep_model()
  for i in range(FLAGS.num_timesteps):
    if not FLAGS.use_rnn:
      timestep_model = create_timestep_model('_'+str(i))

    timestep = timestep_model(intermediates[-1])
    if FLAGS.use_residual:
      timestep += intermediates[-1]
    intermediates.append(timestep)

  decoder = tf.keras.Sequential(name="decoder")
  for _ in range(FLAGS.decoder_layers-1):
    decoder.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  decoder.add(tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("decoder", decoder.layers)

  decoder_counter = tf.keras.Sequential(name="decoder-counter")
  for _ in range(FLAGS.decoder_layers-1):
    decoder_counter.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
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

  adver_decoder = tf.keras.Sequential(
    [
      tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x)),
    ], name="adver_decoder"
  )
  for _ in range(FLAGS.decoder_layers - 1):
    adver_decoder.add(tf.keras.layers.Conv2D(FLAGS.encoded_size, 3, activation=leak_relu(), padding='same',
                                             kernel_regularizer=tf.keras.regularizers.l2(1)))
  adver_decoder.add(
    tf.keras.layers.Conv2D(1, 3, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1)))
  print("adver_decoder", adver_decoder.layers)

  return encoder, intermediates, decoder, adver_decoder, decoder_counter, model, discriminator
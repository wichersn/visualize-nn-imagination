import tensorflow as tf
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_timesteps', 3, '')
flags.DEFINE_integer('encoded_size', 8, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('timestep_layers', 3, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_integer('decoder_counter_layers', 2, 'Only used for the count cells task.')
flags.DEFINE_integer('decoder_counter_strides', 2, 'Only used for the count cells task.')
flags.DEFINE_integer('use_residual', 1, '')
flags.DEFINE_integer('use_rnn', 1, '')
flags.DEFINE_float('dropout_rate', 0.0, '')

leak_relu = tf.keras.layers.LeakyReLU

def get_patches(x, kernel_size):
  patches = tf.image.extract_patches(x,
                          sizes=[1, kernel_size, kernel_size, 1],
                          strides=[1, 1, 1, 1],
                          rates=[1, 1, 1, 1],
                          padding='SAME')
  patches = tf.reshape(patches, patches.shape[:3]+[kernel_size, kernel_size, x.shape[-1]])
  return patches

def get_dists(patches_shape):
  kernel_size = patches_shape[-2]
  single_axis_dist = tf.range(kernel_size, dtype=tf.float32) - kernel_size // 2
  dists = tf.meshgrid(single_axis_dist, single_axis_dist)
  for i in range(len(dists)):
    dists[i] = tf.expand_dims(dists[i], 2)
  dists = tf.concat((dists[0], dists[1]), 2)
  dists = tf.reshape(dists, [1,1,1]+dists.shape)
  dists = tf.tile(dists, patches_shape[:3]+[1,1,1])
  return dists

def reshape_atten_val(val):
  return tf.reshape(val, [val.shape[0]*val.shape[1]*val.shape[2], val.shape[3]*val.shape[4], val.shape[5]])

class VisionSelfAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, key_dim, kernel_size=3):
    super(VisionSelfAttention, self).__init__()
    self.kernel_size = kernel_size
    self.atten_layer = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)

  def call(self, x):
    print("printing in call fn")
    tf.print("tf prinintg in call")
    x = tf.reshape(x, [FLAGS.batch_size]+x.shape[1:])
    patches = get_patches(x, self.kernel_size)

    dists = get_dists(patches.shape)

    key_val_input = tf.concat([patches, dists], -1)

    queries = x
    queries = tf.reshape(queries, [queries.shape[0] * queries.shape[1] * queries.shape[2], 1, queries.shape[3]])

    key_val_input_reshape = reshape_atten_val(key_val_input)

    atten_out = self.atten_layer(queries, key_val_input_reshape, key_val_input_reshape)

    atten_out = tf.reshape(atten_out, x.shape[:-1] + [atten_out.shape[-1]])

    return atten_out

def create_timestep_model(name=''):
  timestep_model = tf.keras.Sequential(name="timestep_model"+name)
  for _ in range(FLAGS.timestep_layers):
    timestep_model.add(VisionSelfAttention(4,4))
    timestep_model.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))
  print("timestep_model", timestep_model.layers)
  return timestep_model

def add_decoder_layers(decoder, decoder_layers, encoded_size=None):
  if not encoded_size:
    encoded_size = FLAGS.encoded_size
  for _ in range(decoder_layers-1):
    decoder.add(VisionSelfAttention(4,4))
    decoder.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))

def get_stop_grad_dec(decoder_layers, name, encoded_size=None):
  decoder = tf.keras.Sequential(
    [
      tf.keras.layers.Lambda(lambda x: tf.keras.backend.stop_gradient(x)),
    ], name=name
  )
  add_decoder_layers(decoder, decoder_layers, encoded_size)
  decoder.add(VisionSelfAttention(4,4))
  print(name, decoder.layers)
  return decoder

def create_models():
  input_shape = [FLAGS.board_size, FLAGS.board_size] + [1, ]
  input_layer = tf.keras.Input(shape=input_shape)

  encoder = tf.keras.Sequential(name="encoder")
  for _ in range(FLAGS.encoder_layers):
    encoder.add(VisionSelfAttention(4,4))
    encoder.add(tf.keras.layers.Dropout(FLAGS.dropout_rate))
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
  add_decoder_layers(decoder, FLAGS.decoder_layers)
  decoder.add(VisionSelfAttention(4,4))
  print("decoder", decoder.layers)

  decoder_counter = tf.keras.Sequential(name="decoder-counter")
  add_decoder_layers(decoder_counter, FLAGS.decoder_counter_layers-1)
  decoder_counter.add(tf.keras.layers.Flatten())
  decoder_counter.add(tf.keras.layers.Dense(1))
  print("decoder_counter", decoder_counter.layers)

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

  adver_decoder = get_stop_grad_dec(FLAGS.decoder_layers, "adver_decoder")

  return encoder, intermediates, decoder, adver_decoder, decoder_counter, model, discriminator
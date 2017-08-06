from tf_utils import *


class ConvOp():
  def __init__(self, W_shape, inputs, prev_res=None, mask=None, activation=True,
    mood_conditional=None, phrase_conditional=None):
    self.inputs = inputs
    in_dim = self.inputs.get_shape()[-1]
    self.W_shape = [W_shape[0], W_shape[1], in_dim, W_shape[2]]
    self.b_shape = W_shape[2]

    self.prev_res = prev_res
    self.mask = mask
    self.activation = activation
    self.mood_conditional = mood_conditional
    self.phrase_conditional = phrase_conditional



  def process(self):

    if self.mood_conditional is not None:
      return self.conditional_cnn()
    else:
      return self.cnn()

  def conditional_cnn(self):
    W_f = get_W(self.W_shape, "v_W", mask_type=self.mask)
    W_g = get_W(self.W_shape, "h_W", mask_type=self.mask)

    U_fm, U_gm = self.process_conditional(self.mood_conditional, name='mood')
    V_fp, V_gp = self.process_conditional(self.phrase_conditional, name='phrase')

    conv_f = tf.nn.conv2d(self.inputs, W_f, strides=[1, 1, 1, 1], padding='SAME')
    conv_g = tf.nn.conv2d(self.inputs, W_g, strides=[1, 1, 1, 1], padding='SAME')

    if self.prev_res is not None:
      conv_f += self.prev_res
      conv_g += self.prev_res

    return tf.multiply(tf.tanh(conv_f + U_fm + V_fp), tf.sigmoid(conv_g + U_gm + V_gp))

  def process_conditional(self, conditional, name='P'):
    conditional_size = int(conditional.get_shape()[1])
    U_f = get_W([conditional_size, self.W_shape[3]], "v_" + name)
    U_fm = tf.matmul(conditional, U_f)
    U_g = get_W([conditional_size, self.W_shape[3]], "h_" + name)
    U_gm = tf.matmul(conditional, U_g)
    b_fm_shape, b_gm_shape = tf.shape(U_fm), tf.shape(U_gm)

    U_fm = tf.reshape(U_fm, (b_fm_shape[0], 1, 1, b_fm_shape[1]))

    U_gm = tf.reshape(U_gm, (b_gm_shape[0], 1, 1, b_gm_shape[1]))

    return U_fm, U_gm

  def cnn(self):
    W = get_W(self.W_shape, "W", mask_type=self.mask)
    b = get_W(self.b_shape, "b", b_initializer)
    conv = tf.nn.conv2d(self.inputs, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_b = tf.add(conv, b)
    if self.activation:
      output = tf.nn.relu(conv_b)
    else:
      output = conv_b

    return output


class MCPixelCNN(object):
  def __init__(self, params):
    self.X = tf.placeholder(tf.float32,
      shape=[None, params.img_height, params.img_width, params.channel])
    self.h_M = tf.placeholder(tf.float32, shape=[None, params.mood_dim])
    self.h_P = tf.placeholder(tf.float32, shape=[None, params.phrase_dim])
    self.params = params
    self.process()
    self.optimize()

  def apply_conv(self, V_l, layer, filter_h, filter_w, stype='V', mask_type='a', prev_res=None,
    use_residual=False):

    with tf.variable_scope("%s_i_%s" % (stype, layer)):
      V = ConvOp([filter_h, filter_w, self.params.f_map], V_l, mask=mask_type, prev_res=prev_res,
        mood_conditional=self.h_M, phrase_conditional=self.h_P)

    with tf.variable_scope("%s_i1_%s" % (stype, layer)):
      V_1 = ConvOp([1, 1, self.params.f_map], V, mask=mask_type)
      if use_residual:
        V_1 += V_l

    return V, V_1

  def process(self):
    V_i, H_i = self.X, self.X
    filter_sizes = [3] + [7] * (self.params.layers - 1)
    mask_types = ['a'] + ['b'] * (self.params.layers - 1)
    use_residual = [False] + [True] * (self.params.layers - 1)

    for i in range(self.params.layers):
      filter_size = filter_sizes[i]
      mask_type = mask_types[i]
      use_residual = use_residual[i]

      V_i, V_1 = self.apply_conv(V_i, i, filter_size, filter_size, 'V', mask_type)
      H_curr, H_1 = self.apply_conv(H_i, i, 1, filter_size, 'H', mask_type, prev_res=V_1,
        use_residual=use_residual)

      H_i = H_1

    self.prepare_fcs(H_i)

    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2,
      tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))

    self.samples = tf.reshape(
      tf.multinomial(tf.nn.softmax(self.fc2), num_samples=1, seed=100), tf.shape(self.X))
    self.preds = tf.reshape(tf.argmax(tf.nn.softmax(self.fc2), dimension=tf.rank(self.fc2) - 1),
      tf.shape(self.X))

  def prepare_fcs(self, H_i):
    with tf.variable_scope("fc_1"):
      fc1 = ConvOp([1, 1, self.params.f_map], H_i, mask='b')
    color_dim = 256
    with tf.variable_scope("fc_2"):
      self.fc2 = ConvOp([1, 1, self.params.channel * color_dim], fc1, mask='b',
        activation=False)
      self.fc2 = tf.reshape(self.fc2, (-1, color_dim))


  def optimize(self):
    self.trainer = tf.train.RMSPropOptimizer(1e-3)
    self.gradients = self.trainer.compute_gradients(self.loss)

    self.clipped_gradients = [(tf.clip_by_value(_[0], self.params.grad_clip, self.params.grad_clip), _[1]) for _ in gradients]
    self.opt = self.trainer.apply_gradients(self.clipped_gradients)


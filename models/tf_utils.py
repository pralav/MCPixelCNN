import tensorflow as tf
import numpy as np
import os

INITIALIZER = tf.contrib.layers.xavier_initializer()

def zeros_initializer():
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.zeros(shape,dtype=dtype)
  return _initializer
b_initializer = zeros_initializer()

def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u

def get_W(name, weights_shape, initializer=INITIALIZER, mask_type=None):
  weights = tf.get_variable(name, weights_shape, tf.float32, initializer)

  if mask_type is not None:
    filter_h,filter_w=weights_shape[0],weights_shape[1]
    assert filter_h % 2 == 1 and filter_w % 2 == 1, \
      "kernel height and width should be odd number"
    center_h = filter_h // 2
    center_w = filter_w // 2
    mask_type = mask_type.lower()
    mask = np.ones(weights_shape, dtype=np.float32)
    mask[center_h, center_w + 1:, :, :] = 0.
    mask[center_h + 1:, :, :, :] = 0.
    if mask_type == 'a':
      mask[center_h, center_w, :, :] = 0.
    weights *= tf.constant(mask, dtype=tf.float32)

  return weights


def create_model_paths(params):
  ckpt_full_path = os.path.join(params.ckpt_path, "%s" % (params.model))
  if not os.path.exists(ckpt_full_path):
    os.makedirs(ckpt_full_path)
  params.ckpt_file = os.path.join(ckpt_full_path, "model%s.ckpt"%params.epochs)

  params.samples_path = os.path.join(params.samples_path, "samples","%s" % (params.model))
  if not os.path.exists(params.samples_path):
    os.makedirs(params.samples_path)

  if tf.gfile.Exists(params.summary_path):
    tf.gfile.DeleteRecursively(params.summary_path)
  tf.gfile.MakeDirs(params.summary_path)

  return params


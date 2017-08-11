import random

import numpy as np
import os
import scipy.misc
from datetime import datetime

def generate_samples(sess, X, phrase_h, mood_h, phrase_embs, mood_embs, pred, params, suff):
  n_row, n_col = 5,5
  samples = np.zeros((n_row*n_col, params.img_height, params.img_width, params.channel), dtype=np.float32)
  phrase_emb=np.tile(random.choice(phrase_embs),(n_row*n_col,1))
  mood_emb=np.tile(random.choice(mood_embs),(n_row*n_col,1))
  for i in xrange(params.img_height):
    for j in xrange(params.img_width):
      for k in xrange(params.channel):
        feed_dict = {X:samples, phrase_h:phrase_emb,mood_h:mood_emb}
        next_sample = sess.run(pred, feed_dict=feed_dict)
        samples[:, i, j, k] = next_sample[:, i, j, k]

  save_images(samples, n_row, n_col, params, suff)



def save_images(samples, n_row, n_col, conf, suff):
  images = samples

  images = images.reshape((n_row, n_col, conf.img_height, conf.img_width, conf.channel))
  images = images.transpose(1, 2, 0, 3, 4)
  images = images.reshape((conf.img_height * n_row, conf.img_width * n_col, conf.channel))

  filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+suff+".jpg"
  scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))


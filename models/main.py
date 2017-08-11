import tensorflow as tf
import argparse
from mc_pixxelcnn import MCPixelCNN
# from autoencoder import *
from img_utils import *
from tf_utils import *

import cPickle

def train(params, img_data, phrase_data, mood_data):

  model = MCPixelCNN(params)
  saver = tf.train.Saver(tf.trainable_variables())

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    if os.path.exists(params.ckpt_file):
      saver.restore(sess, params.ckpt_file)
    pointer = 0
    for i in range(params.epochs):
      for batch in range(params.num_batches):
        batch_X = img_data[params.batch_size * batch : params.batch_size * (batch + 1)]
        batch_hM = phrase_data[params.batch_size * batch : params.batch_size * (batch + 1)]
        batch_hP = mood_data[params.batch_size * batch : params.batch_size * (batch + 1)]

        _, loss = sess.run([model.opt, model.loss], feed_dict={model.X:batch_X,model.h_M:batch_hM,model.h_P:batch_hP})
        if batch%10==0:
          print "Epoch: %d, Loss: %f"%(i, loss)

      if i>0 and i%10 == 0:
        saver.save(sess, params.ckpt_file)
        generate_samples(sess, model.X, model.h_P, model.h_M, model.preds, params, "")



if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('-l','--layers', type=int, default=12, help="No. of layers")
  parser.add_argument('-fs','--fs', type=int, default=32,help="fs")
  parser.add_argument('-E','--epochs', type=int, default=50, help="Epochs")
  parser.add_argument('-B','--batch_size', type=int, default=100,  help="Batch Size")
  parser.add_argument('-H','--img_height', type=int, default=32, help="Img height")
  parser.add_argument('-w','--img_width', type=int, default=32, help="Img Width")
  parser.add_argument('-C','--channel', type=int, default=3, help="Channels")
  parser.add_argument('-g','--grad_clip', type=int, default=1, help="Gradient clipping")
  parser.add_argument('-m','--model', type=str, default='', help="Model name")
  parser.add_argument('-d','--data_path', type=str, default='data', help="Data path")
  parser.add_argument('-c','--ckpt_path', type=str, default='mcpixelcnn',help="Checkpoints path")
  parser.add_argument('-s','--samples_dir', type=str, default='samples',help="Sample Directory")
  parser.add_argument('-S','--summary_path', type=str, default='logs',help="Summary Directory")

  args = parser.parse_args()


  data=cPickle.load(open(args.data_path))
  imgs,phrases,moods=zip(*data)
  labels = data[0][1]
  data = data[0][0].astype(np.float32)
  imgs[:,0,:,:] -= np.mean(data[:,0,:,:])
  imgs[:,1,:,:] -= np.mean(data[:,1,:,:])
  imgs[:,2,:,:] -= np.mean(data[:,2,:,:])
  imgs = np.transpose(data, (0, 2, 3, 1))


  args.n_batches = data.shape[0] // args.batch_size

  args = create_model_paths(args)
  args.conditional = True
  train(args,imgs,phrases,moods)



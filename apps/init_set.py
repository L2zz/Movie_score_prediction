import os
import tensorflow as tf

def init():
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.19
  session = tf.Session(config=config)

  return session

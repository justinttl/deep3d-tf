import tensorflow as tf

def loss(scale, fuse='sum', get_softmax=False, method='multi2', upsample=1, tan=False):


    left = tf.Variable(tf.float32,name='left')

    left0 = tf.Variable(tf.float32,name='left0')

    up, _ =


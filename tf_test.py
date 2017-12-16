import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
with tf.Session() as sess:
    result =  sess.run(node1)
    print(result)

from __future__ import print_function
import tensorflow as tf

a = tf.get_variable("a", dtype=tf.float32, initializer=tf.constant(0.0))
b = tf.get_variable("b", dtype=tf.float32, initializer=tf.constant(0.0))
x = tf.placeholder(tf.float32)
linear_model = a*x + b
y = tf.placeholder(tf.float32)


loss = tf.reduce_sum(tf.square(linear_model-y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train=[0, -1, -2, -3]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        sess.run(train, feed_dict={x:x_train, y:y_train})

    curr_a, curr_b, curr_loss = sess.run([a, b, loss], feed_dict={x:x_train, y:y_train})
    print("a:%s, b:%s, loss:%s", curr_a, curr_b, curr_loss)



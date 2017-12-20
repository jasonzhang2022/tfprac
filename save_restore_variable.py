from __future__ import print_function
import tensorflow as tf

'''
variable can be saved by tf.train.Saver().
Varible only has value in session. So varible is saved throuh session.
Variable can be saved selectively with a different name.
'''

v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.random_normal_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2= v2.assign(v2-1)

#only save v2 as v3
saver = tf.train.Saver({"v3": v2})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #run the operation where tensor comes from
    inc_v1.op.run()
    dec_v2 = v2.op.run()

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("model saved in file %s" % save_path)

v3 = tf.get_variable("v3", shape=[3], initializer=tf.zeros_initializer)
print("------------ restore variables--------------")
with tf.Session() as sess:
    saver.restore(sess, "/tmp/model.ckpt")
    for x in tf.global_variables():
        #v2 is initialized, but v1 is not initialized
        print (x.name, "is initialized or not: ", sess.run(tf.is_variable_initialized(x)))




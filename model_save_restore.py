from __future__ import print_function
import tensorflow as tf
import shutil

export_dir = "/tmp/test"
shutil.rmtree(export_dir)
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

#construct a session and its graph.
#where we don't use default grapph
with tf.Session(graph=tf.Graph()) as sess:
    v1 = tf.constant([[3,4,5]], dtype=tf.float32, name="v1")
    v2 = tf.get_variable("v2",  shape=[3,1], dtype=tf.float32, initializer=tf.random_normal_initializer)
    v3 = tf.matmul(v1, v2)
    sess.run(tf.global_variables_initializer())
    builder.add_meta_graph_and_variables(sess, #save the graph associated with sess
                                         [tf.saved_model.tag_constants.TRAINING], #a list of tags to identify the graph
                                         signature_def_map={}, # a list of input and output
                                       #  assets_collection=[] #what asset is?
                                         )
#we can save another graph here
builder.save()



with tf.Session(graph = tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], export_dir)
    # only v2 variable is loaded. v1 is constant is not loaded
    for v in tf.global_variables() :
        print (v.name, "is initialized :", sess.run(tf.is_variable_initialized(v)))

from __future__ import print_function
import tensorflow as tf
'''
Graph has graph structure and graph context(graph collections).

tensorflow has two phases: graph building phase and session execution phase.

In execution phase, graph need to retrieve value and store session to graph collections.

graph may have may disconnected components.

In building phase, operation is added to graph and return a tensor. Tensor can be used to connect operation
to other node.

'''
with tf.name_scope("outer"):
    c2=tf.constant(3, name="c")
    print("tensor name", c2.name)
    #this is like inspect the graph where this tensor comes from
    print("operator name", c2.op.name)
    #what graph this tensor belongs to.
    print(c2.graph)
    print (c2.op.graph)
    #we can find out default graph.
    assert c2.graph is tf.get_default_graph()
    print(len(tf.get_default_graph().get_operations()))


#oprations can be placed to different device like variable

'''
to execute the graph, client(this python script) need to establish a session to TensorFlow server.
by default the server is local server.
But target is needed if you want to establish connection to remote server

Graph to be executed need passing to session. Default graph is used if no one is specified.

Other execution options: like soft device placement, cluster definition, optimizer option, memory policy can be specificied
'''
x_prime = tf.constant([[37.0, -23.0], [1.0, 4.0]])
x = tf.placeholder(dtype=tf.float32, shape=[2,2])
w = tf.Variable(tf.random_uniform([2,2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # here [y, output] are called fetch: that values we'd like to retrieve as an result of
    # execution
    # here feed_dict: is necessary input feed so Tensorflow has enough information to compute
    # the fetch. Once a fetch is computed, it will not be computed again.
    y_val, output_val = sess.run([y, output], feed_dict={x : x_prime.eval()})
    print ("y_val ", y_val)
    print ("output ", output_val)

    '''
    session.run also accepts options like trace level, whetther to capture intermdiate data or 
    meta data or not
    '''



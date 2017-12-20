from __future__ import print_function
import tensorflow as tf

'''
variable can have 
    type, 
    shape, 
    initializer
    
It is persisted cross session

initializer can initialize the tensor with same shape
'''

'''
a variable named as test,
it is 3-dimension 
it uses default initializer: glorot_uniform_initializer
it is added to global collection and trainable collection.
'''
my_variable = tf.get_variable("test", [2,3,4])

# specify datatype, and initializer explicitly
my_variable1 = tf.get_variable("test1", [2,3,4], dtype=tf.int32, initializer=tf.zeros_initializer)

#initialize variable from another tensor, so don't specify shape.
#the shape follows the shape of the tensor.
my_variable2 = tf.get_variable("test2", dtype=tf.int32, initializer=tf.constant([23,42]))

#variable by default is added to global collection and trainable collection
#here we specify the variable is added only to global collection
my_variable3 = tf.get_variable("test3", shape=(), collections=[tf.GraphKeys.GLOBAL_VARIABLES])
#don't add variable from trainable collection
my_variable4 = tf.get_variable("test4", trainable=False, initializer=my_variable3.initialized_value()+1)
#explictly add variable to one of collection
tf.add_to_collection("one_local_collection", my_variable4)

#variable can be placed to a specific
'''
with tf.device("/device:GPU:1")
  v = tf.get_variable("v", 1)
  
with tf.device(tf.train.replica_device_setter(cluster=cluster=spec))
 v = tf.get_variable("v", shape=[20, 20])
'''

#variable can be intialized with global initializer or with variable specific intiializer
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(my_variable))

    print("\n report unitialized variable. Should show any empty list")
    print(sess.run(tf.report_uninitialized_variables()))

    #variable 4 is initialized through global variable initializer.
    print(sess.run(my_variable4))
    #it is initializer explicitly again
    #since its value is calculated from that of variable3. So its value doesn't change in this session
    sess.run(my_variable4.initializer)
    print(sess.run(my_variable4))


'''
use variable in tensor flow: treat it as tensor
'''
#w is tensor, not variable. so its value is not persisted cross session
w = my_variable3 +1
v = tf.get_variable("v", initializer=my_variable3.initialized_value() +1)
assignment = v.assign(1)
with tf.Session() as sess:
    #operation can run in default session
    tf.global_variables_initializer().run()
    assignment.eval()

'''
variable can be put into scope:
why we need this:
    When a function uses some/create variables, sometime we don't want to share global variable when
    this function is called repeated.
    Instead, caller can define a variable scope
'''

with tf.variable_scope("outer"):
    scoped = tf.get_variable("inner", shape=(), dtype=tf.int32)
    print("variable name: ", scoped.name)



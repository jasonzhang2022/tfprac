import tensorflow as tf
'''
tensor has
    datatype
    shape: rank
'''

'''
 tf.rank(tensor): this returns a opreation which can be evaluated in session
 tensor doesn't have rank property
'''

d3tensor = tf.zeros([3,4,5], tf.int32)

# shape can be decide at build time. don't need to figure out its shape at run time
d3tensor1 = tf.ones(tf.shape(d3tensor))
print " ---------------demo tensor.shape property"
print(d3tensor1.shape) #(3,4,5). Tensor has shape property
rank_op = tf.rank(d3tensor)

#slice
column = d3tensor[2,3,:]
#access a single element
single_value = d3tensor[2,3,4]

#reshape
reshaped = tf.reshape(d3tensor1, [12, -1])



print "------------demo tf.rank, tf.shape operation, tensor slicing, reshape"
with tf.Session() as sess:
    rank = sess.run(rank_op)
    print(rank)
    print (sess.run(column))
    print(sess.run(single_value))
    #tf shape is an operation, must be evaluated
    #output is [3 4 5] which is a list
    #where tensor.shape outputs a tuple(3,4,5)
    #this operation can be used to retrieve dynamic shape and use it
    # to build execution graph
    print (sess.run(tf.shape(d3tensor1)))
    #output is [3 4 5] which is a
    print(sess.run(d3tensor1[:, 3,4]))
    print ("reshaped ---")
    print (sess.run(reshaped))


# evaluate tensor out of session
print " --------------demo tensor.eval()"
sess = tf.Session()
constant = tf.constant([1,2,3])
plus = constant * constant
print plus.eval(session=sess)

print "---------------evaluate with feed_dict"
p = tf.placeholder(tf.float32)
t= p + 1.0
print t.eval(session=sess, feed_dict={p:2.0})
sess.close()


print " -------- demo tf.Print which is like tie operator in terminal"
random = tf.random_uniform([2,3])
random = tf.Print(random, [random])
result = random + 1
with tf.Session() as sess:
    sess.run(result)

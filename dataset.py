from __future__ import print_function
import tensorflow as tf

'''
Dataset can be initialized from 
    tensor  
    from disk(numpy data, TFDatasetRecord, Text)
Structure:
    viewed as a stream of element. Each element has multiple components
'''

'''
one shot initializer from dataset. Don't need to run iterator.initializer
'''
dataset = tf.data.Dataset.range(100)
#iterator is from dataset. No initializer needs calling
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))



'''
initializable iterator: 
iterator is from a dataset and associated with dataset. 
neeed to run iterator initializer explicitly, can pass feed_dict
'''
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
print(dataset1.output_types) # has output_types, and output_shapes propertie4s
print(dataset1.output_shapes) # the shape of each element. not the shape of input tensor

dataset2 = tf.data.Dataset.from_tensor_slices( (tf.random_uniform([4]), tf.random_uniform([4,5], maxval=100, dtype=tf.int32)))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))


iterator = dataset3.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    # this is like the object spread operation in ES6
    value10, (singleValue, value5)= sess.run(next_element)
    assert(len(value10)) is 10
    assert(len(value5)) is 5
    rank = sess.run(tf.rank(singleValue))
    print(rank)


max_value = tf.placeholder(tf.int64, shape=[])
dataset1 = tf.data.Dataset.range(max_value)
iterator = dataset1.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    #iterator is from dataset, we can pass feed_dict to initializer
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    print (sess.run(next_element))


'''
reinitializable iterator:
iterator is bound to dataset, can switch/reinitialize the iterator
'''
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x+tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)
iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

#an operation associate iterator with dataset
training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
    sess.run(training_init_op)
    #from training set
    print(sess.run(next_element))

    sess.run(validation_init_op)
    #from validation set
    print(sess.run(next_element))


'''
reintializable iterator can associate iterator with one dataset through initializer 
before dataset loop

feedable iterator: can associate data source for each next_element operation
    
'''
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()
with tf.Session() as sess:
    #turn iterator into handle: placeholder
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    sess.run(validation_iterator.initializer)

    #from training
    sess.run(next_element, feed_dict={handle: training_handle})

    #from validation: each operation specify where the data source is from
    sess.run(next_element, feed_dict={handle: validation_handle})

'''
dataset can be readed from numpy arrays through np.load
dataset can be loaded from TFRecordDataset or TextLineDataset
'''

'''
dataset can be mapped, batched, repated, shuffled

'''
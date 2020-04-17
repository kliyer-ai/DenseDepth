import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework.ops import get_gradient_function

# a = tf.add(1, 2, name="Add_these_numbers")
# b = tf.multiply(a, 3, name='mult')

# mult = tf.get_default_graph().get_operation_by_name('mult')
# print(get_gradient_function(mult))  # <function _MulGrad at 0x7fa29950dc80>

# c = tf.where([True], name='where')
# where = tf.get_default_graph().get_operation_by_name('where')
# print(get_gradient_function(where)) 


# c = tf.image.crop_to_bounding_box
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
arr = np.array([1,2,3,4])

with tf.Session() as sess: 
    arr = sess.run(tf.reshape(arr, [-1, 1, 1, 1]))
    arr = sess.run(tf.tile(arr, [1, 3, 3, 3]))
    print(arr)
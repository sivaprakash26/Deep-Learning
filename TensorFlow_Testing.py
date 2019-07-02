# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:44:13 2019

@author: siva1
"""

import tensorflow as tf

a = tf.constant(3, name='Constant_a')
b = tf.constant(6, name='Constant_b')
c = tf.constant(10, name='Constant_c')
d = tf.constant(5, name='Constant_d')

mul = tf.multiply(a, b, name = 'MUL')
div = tf.div(c, d, name = 'DIV')
add_n = tf.add_n([mul, div], name = 'ADD_N')

sess = tf.Session()

print(sess.run(add_n))

writer = tf.summary.FileWriter(r'C:\Users\siva1\Machine Learning\DeepLearning\Testing_example', sess.graph)
writer.close()
sess.close()



import tensorflow as tf

a = tf.constant([3, 4], name='a')
b = tf.constant([4, 5], name='b')
add_op = a + b

with tf.Session() as session:
    print(session.run(add_op))

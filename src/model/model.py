import tensorflow.compat.v1 as tf #type: ignore
tf.disable_v2_behavior()

# Helper functions
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

# Placeholders
x = tf.placeholder(tf.float32, [None, 66, 200, 3])
y_true = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

# Model definition
W1 = weight_variable([5, 5, 3, 24])
b1 = bias_variable([24])
h1 = tf.nn.relu(conv2d(x, W1, 2) + b1)

W2 = weight_variable([5, 5, 24, 36])
b2 = bias_variable([36])
h2 = tf.nn.relu(conv2d(h1, W2, 2) + b2)

W3 = weight_variable([5, 5, 36, 48])
b3 = bias_variable([48])
h3 = tf.nn.relu(conv2d(h2, W3, 2) + b3)

W4 = weight_variable([3, 3, 48, 64])
b4 = bias_variable([64])
h4 = tf.nn.relu(conv2d(h3, W4, 1) + b4)

W5 = weight_variable([3, 3, 64, 64])
b5 = bias_variable([64])
h5 = tf.nn.relu(conv2d(h4, W5, 1) + b5)

# Flatten
flat = tf.reshape(h5, [-1, 1152])

# Fully connected layers
W_fc1 = weight_variable([1152, 1164])
b_fc1 = bias_variable([1164])
fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
fc1 = tf.nn.dropout(fc1, keep_prob)

W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])
fc2 = tf.nn.relu(tf.matmul(fc1, W_fc2) + b_fc2)
fc2 = tf.nn.dropout(fc2, keep_prob)

W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])
fc3 = tf.nn.relu(tf.matmul(fc2, W_fc3) + b_fc3)
fc3 = tf.nn.dropout(fc3, keep_prob)

W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])
fc4 = tf.nn.relu(tf.matmul(fc3, W_fc4) + b_fc4)
fc4 = tf.nn.dropout(fc4, keep_prob)

W_out = weight_variable([10, 1])
b_out = bias_variable([1])

# Output angle (scaled)
y_pred = tf.multiply(tf.atan(tf.matmul(fc4, W_out) + b_out), 2)

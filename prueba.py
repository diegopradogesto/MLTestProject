import tensorflow as tf
import numpy as np

# x and y are placeholders for our training data
x = tf.placeholder(tf.float32, [None, 3], name = "input")
y = tf.placeholder(tf.float32, [None, 1], name = "target")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
W1 = tf.Variable(tf.truncated_normal([3, 1], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.truncated_normal([1]), name = 'b1')
# Our model of y = a*x + b
y_model = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# Our error is defined as the square of the differences
error = tf.reduce_mean(tf.square(y - y_model))
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    avg_cost = 0
    for i in range(10000):
        x1_value = np.random.rand()
        x2_value = np.random.rand()
        x3_value = np.random.rand()
        y_value = [[x1_value * 2 - x2_value * 5 + x3_value * 3 + 10]]
        x_value = [[x1_value, x2_value, x3_value]]
        _, err = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
        avg_cost += err
        if (i + 1) % 10 == 0:
            print("Epoch: {epoch} - cost = {cost}".format(epoch = i + 1, cost = err))

            w_value = session.run(W1)
            b_value = session.run(b1)
            estimation = w_value[0] * x1_value + w_value[1] * x2_value + w_value[2] * x3_value + b_value
            print("Difference: {}".format((y_value - estimation)*(y_value - estimation)))

            print("Predicted model: {constx}*{valx} + {consty}*{valy} + {constz}*{valz} + {b} = {result} <> {y_value}".format(
                valx=x1_value, valy=x2_value, valz=x3_value, b=b_value[0], result = estimation, y_value = y_value,
                constx=w_value[0], consty=w_value[1], constz=w_value[2]))

            print("Estimation: {est} - {estt}".format(est = estimation, estt = session.run(y_model, feed_dict={x: x_value})))
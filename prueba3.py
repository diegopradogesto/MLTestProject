import tensorflow as tf
import numpy as np

# x and y are placeholders for our training data
x = tf.placeholder(tf.float32, [None, 2], name = "input")
y = tf.placeholder(tf.float32, [None, 1], name = "target")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
W1 = tf.Variable(tf.truncated_normal([2, 1], stddev = 0.03), name = 'W1')
b1 = tf.Variable(tf.truncated_normal([1]), name = 'b1')
# Our model of y = a*x + b
y_model = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))

# Our error is defined as the square of the differences
error = tf.reduce_mean(tf.square(y - y_model))
# The Gradient Descent Optimizer does the heavy lifting
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()
inputs = [[1.0, 1.25], [1.0, 0.25], [1.331, 1.25], [1.331, 0.25], [1.728, 1.25], [1.728, 0.25], [2.197, 1.25], [2.197, 0.25], [2.744, 1.25], [2.744, 0.25], [3.375, 1.25], [3.375, 0.25], [4.096, 1.25], [4.096, 0.25], [4.913, 1.25], [4.913, 0.25], [5.832, 1.25], [5.832, 0.25], [6.859, 1.25], [6.859, 0.25], [8.0, 1.25], [8.0, 0.25], [9.261, 1.25], [9.261, 0.25], [10.648, 1.25], [10.648, 0.25], [12.167, 1.25], [12.167, 0.25], [13.824, 1.25], [13.824, 0.25], [15.625, 1.25], [15.625, 0.25], [17.576, 1.25], [17.576, 0.25], [19.683, 1.25], [19.683, 0.25], [21.952, 1.25], [21.952, 0.25], [24.389, 1.25], [24.389, 0.25], [27.0, 1.25], [27.0, 0.25], [29.791, 1.25], [29.791, 0.25], [32.768, 1.25], [32.768, 0.25], [35.937, 1.25], [35.937, 0.25], [39.304, 1.25], [39.304, 0.25], [42.875, 1.25], [42.875, 0.25], [46.656, 1.25], [46.656, 0.25], [50.653, 1.25], [50.653, 0.25], [54.872, 1.25], [54.872, 0.25], [59.319, 1.25], [59.319, 0.25], [64.0, 1.25], [64.0, 0.25], [68.921, 1.25], [68.921, 0.25], [74.088, 1.25], [74.088, 0.25], [79.507, 1.25], [79.507, 0.25], [85.184, 1.25], [85.184, 0.25], [91.125, 1.25], [91.125, 0.25], [97.336, 1.25], [97.336, 0.25], [103.823, 1.25], [103.823, 0.25], [110.592, 1.25], [110.592, 0.25], [117.649, 1.25], [117.649, 0.25], [125.0, 1.25], [125.0, 0.25], [132.651, 1.25], [132.651, 0.25], [140.608, 1.25], [140.608, 0.25], [148.877, 1.25], [148.877, 0.25], [157.464, 1.25], [157.464, 0.25], [166.375, 1.25], [166.375, 0.25], [175.616, 1.25], [175.616, 0.25], [185.193, 1.25], [185.193, 0.25], [195.112, 1.25], [195.112, 0.25], [205.379, 1.25], [205.379, 0.25], [216.0, 1.25], [216.0, 0.25], [226.981, 1.25], [226.981, 0.25], [238.328, 1.25], [238.328, 0.25], [250.047, 1.25], [250.047, 0.25], [262.144, 1.25], [262.144, 0.25], [274.625, 1.25], [274.625, 0.25], [287.496, 1.25], [287.496, 0.25], [300.763, 1.25], [300.763, 0.25], [314.432, 1.25], [314.432, 0.25], [328.509, 1.25], [328.509, 0.25], [343.0, 1.25], [343.0, 0.25], [357.911, 1.25], [357.911, 0.25], [373.248, 1.25], [373.248, 0.25], [389.017, 1.25], [389.017, 0.25], [405.224, 1.25], [405.224, 0.25], [421.875, 1.25], [421.875, 0.25], [438.976, 1.25], [438.976, 0.25], [456.533, 1.25], [456.533, 0.25], [474.552, 1.25], [474.552, 0.25], [493.039, 1.25], [493.039, 0.25], [512.0, 1.25], [512.0, 0.25], [531.441, 1.25], [531.441, 0.25], [551.368, 1.25], [551.368, 0.25], [571.787, 1.25], [571.787, 0.25], [592.704, 1.25], [592.704, 0.25], [614.125, 1.25], [614.125, 0.25], [636.056, 1.25], [636.056, 0.25], [658.503, 1.25], [658.503, 0.25], [681.472, 1.25], [681.472, 0.25], [704.969, 1.25], [704.969, 0.25], [729.0, 1.25], [729.0, 0.25], [753.571, 1.25], [753.571, 0.25], [778.688, 1.25], [778.688, 0.25], [804.357, 1.25], [804.357, 0.25], [830.584, 1.25], [830.584, 0.25], [857.375, 1.25], [857.375, 0.25], [884.736, 1.25], [884.736, 0.25], [912.673, 1.25], [912.673, 0.25], [941.192, 1.25], [941.192, 0.25], [970.299, 1.25], [970.299, 0.25], [1000.0, 1.25], [1000.0, 0.25]]
targets = [[0.19194444444444445], [0.17166666666666666], [0.24416666666666667], [0.1938888888888889], [0.30416666666666664], [0.21666666666666667], [0.3725], [0.24194444444444443], [0.44166666666666665], [0.26944444444444443], [0.5297222222222222], [0.2980555555555556], [0.6366666666666667], [0.3375], [0.7558333333333334], [0.38305555555555554], [0.8886111111111111], [0.43027777777777776], [1.0525], [0.5025], [1.2072222222222222], [0.5708333333333333], [1.373888888888889], [0.6413888888888889], [1.5525], [0.6983333333333334], [1.728611111111111], [0.7672222222222222], [1.9477777777777778], [0.8352777777777778], [2.1977777777777776], [0.9558333333333333], [2.4655555555555555], [1.056111111111111], [2.7525], [1.1388888888888888], [3.0927777777777776], [1.2336111111111112], [3.4025], [1.3283333333333334], [3.7275], [1.4638888888888888], [4.064722222222223], [1.6066666666666667], [4.386666666666667], [1.7427777777777778], [4.8180555555555555], [1.8463888888888889], [5.269722222222223], [1.9713888888888889], [5.744166666666667], [2.093888888888889], [6.318888888888889], [2.3005555555555555], [6.815833333333333], [2.4830555555555556], [7.331944444444445], [2.663611111111111], [7.864444444444445], [2.7819444444444446], [8.351666666666667], [2.9402777777777778], [8.963055555555556], [3.0866666666666664], [9.65], [3.3855555555555554], [10.36138888888889], [3.6158333333333332], [11.100277777777778], [3.7780555555555555], [11.971388888888889], [3.975], [12.723611111111111], [4.1675], [13.496666666666666], [4.4591666666666665], [14.283333333333333], [4.755], [15.005555555555556], [5.035833333333334], [15.953611111111112], [5.215833333333333], [16.95277777777778], [5.453055555555555], [17.98111111111111], [5.678888888888889], [19.041666666666668], [6.088611111111111], [20.259166666666665], [6.435277777777777], [21.317777777777778], [6.772777777777778], [22.39611111111111], [6.965], [23.371111111111112], [7.244166666666667], [24.5225], [7.491944444444444], [25.86638888888889], [8.051944444444445], [27.23472222222222], [8.454166666666667], [28.63722222222222], [8.721111111111112], [30.296666666666667], [9.053888888888888], [31.684722222222224], [9.374166666666667], [33.09916666666667], [9.881388888888889], [34.52777777777778], [10.382222222222222], [35.867222222222225], [10.853333333333333], [37.44444444444444], [11.128333333333334], [39.209722222222226], [11.51], [41.00416666666667], [11.866388888888888], [42.8375], [12.548055555555555], [44.95388888888889], [13.110277777777778], [46.74916666666667], [13.651388888888889], [48.56361111111111], [13.928888888888888], [50.385], [14.360555555555555], [52.03277777777778], [14.731944444444444], [54.248333333333335], [15.635555555555555], [56.49138888888889], [16.261666666666667], [58.76888888888889], [16.65], [61.48388888888889], [17.151944444444446], [63.70305555555556], [17.629722222222224], [65.95277777777778], [18.414722222222224], [68.21583333333334], [19.171944444444446], [70.2286111111111], [19.880555555555556], [72.68777777777778], [20.265], [75.44166666666666], [20.823888888888888], [78.21972222222222], [21.337777777777777], [81.03694444444444], [22.368333333333332], [84.31944444444444], [23.186944444444446], [87.0436111111111], [23.975833333333334], [89.79416666666667], [24.355833333333333], [92.5425], [24.970555555555556], [95.03666666666666], [25.51361111111111], [98.19527777777778], [26.82027777777778], [101.53277777777778], [27.724166666666665], [104.90194444444444], [28.246944444444445], [108.31527777777778], [28.950833333333332], [112.1986111111111], [29.615], [115.47555555555556], [30.741944444444446], [118.77], [31.803055555555556], [121.70111111111112], [32.793055555555554]]

with tf.Session() as session:
    session.run(model)
    avg_cost = 0
    for epoch in range(10000):
        for index in range(len(inputs[0])):
            x_value = [inputs[index]]
            y_value = [targets[index]]
            _, err = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
            avg_cost += err
            if (epoch + 1) % 10 == 0:
                print("Epoch: {epoch} - cost = {cost}".format(epoch = epoch + 1, cost = err))

                w_value = session.run(W1)
                b_value = session.run(b1)
                estimation = w_value[0] * x_value[0][0] + w_value[1] * x_value[0][1] + b_value
                print("Difference: {}".format((y_value[0][0] - estimation)*(y_value[0][0] - estimation)))

                print("Predicted model: {constx}*{valx} + {consty}*{valy} + {b} = {result} <> {y_value}".format(
                    valx=x_value[0][0], valy=x_value[0][1], b=b_value[0], result = estimation, y_value = y_value[0][0],
                    constx=w_value[0], consty=w_value[1]))

                print("Estimation: {est} - {estt}".format(est = estimation, estt = session.run(y_model, feed_dict={x: x_value})))
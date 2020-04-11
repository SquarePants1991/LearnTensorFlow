import tensorflow as tf

x = tf.Variable(1.0, tf.float32)
y = tf.Variable(2.0, tf.float32)
d = tf.constant(2.0)

# x * x + y * y + 2 = 10
pred = tf.add(tf.add(tf.multiply(x, x), tf.multiply(y,y)), d)
# pred = tf.add(x, y)
real = tf.constant(10.0)

loss = tf.multiply(tf.subtract(pred, real), tf.subtract(pred, real))

opt = tf.train.GradientDescentOptimizer(0.01)

train = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(0, 1000):
        sess.run(train)
        print("Loss: " + str(loss.eval()))
        if loss.eval() < 0.0001:
            answer_x = x.eval()
            answer_y = x.eval()
            print('Answer is: ' + str(answer_x) + ', ' + str(answer_y))
            print('Check Answer : ' + str(pred.eval()) + ' = ' + str(real.eval()))
            break



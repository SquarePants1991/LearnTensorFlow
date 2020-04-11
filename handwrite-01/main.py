import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 输入数据，将28 * 28 的图片数据变成一维的784长度的数据，我们有多张数据，所以输入第一维可变
x = tf.placeholder("float", [None, 784])

# x * W + b = y 输出等于输入 * 权重W + 偏移，权重可以理解为输入的每个unit在输出纬度的概率分布
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# tf.matmul(x, W) 得出 None，10维的矩阵，每个数据相当于每个像素在该维度的概率综合
y = tf.nn.softmax(tf.matmul(x, W) + b) # 应该也是 [None, 10] 的矩阵

y_right = tf.placeholder(float, [None, 10])


cross_entropy = -tf.reduce_sum(y_right * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_right:batch_ys})

correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_right, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
res = session.run(accuracy, feed_dict={x: mnist.test.images, y_right: mnist.test.labels})
print(res)


#
# m1 = tf.constant([[1, 1]])
# m2 = tf.constant([[2], [2]])
#
# product = tf.matmul(m1, m2)
#
#
# session = tf.Session()
# result = session.run(product)
# print(result)
#
# session.close()
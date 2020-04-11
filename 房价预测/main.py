import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 取数据
datas = pd.read_csv("data/data0.csv", names=['square', 'price'], dtype=np.float32)

# 规范化数据 (data - mean) / std
processed_datas = datas.apply(lambda col: (col - col.mean()) / col.std())

# 输入数据准备
one_df = pd.DataFrame(np.ones(len(processed_datas['square'])))
train_data = pd.concat([one_df[0], processed_datas['square']], axis=1)
labeled_data = np.array(processed_datas['price']).reshape((len(processed_datas['price']), 1))

with tf.name_scope("Input"):
    # 构建训练模型 price = w0 * b + w1 * square = wT * data
    # len(processed_datas['square']), 2
    input_data = tf.placeholder(tf.float32, name="input-data")
    label_data = tf.placeholder(tf.float32, name="labeled-data")

with tf.name_scope("Weights"):
    # 2, 1
    W = tf.Variable([[10.0], [10.0]], tf.float32)

with tf.name_scope("pred"):
    pred = tf.matmul(input_data, W)

with tf.name_scope("loss"):
    # tf.matmul(pred, labeled_data, transpose_a=True) 相当于向量的点乘，所有向量元素积之和
    # 最后除以总样本数去平均， 就是最小二乘法
    loss = 1 / (2 * len(processed_datas['square'])) * tf.matmul((pred - label_data), (pred - label_data), transpose_a=True)

with tf.name_scope("trainer"):
    opt = tf.train.GradientDescentOptimizer(0.01)
    train_op = opt.minimize(loss)

final_w_val = None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with tf.summary.FileWriter("./summary/data.sum", sess.graph) as writer:
        for epoch in range(0, 1000):
            sess.run(train_op, feed_dict={input_data: train_data, label_data: labeled_data})
            loss_val = sess.run(loss, feed_dict={input_data: train_data, label_data: labeled_data})
            final_w_val = sess.run(W, feed_dict={input_data: train_data, label_data: labeled_data})


# matlibplot可视化数据
print("train finish, loss: {0} price = {1} + {2} * square".format(loss_val, final_w_val[0][0], final_w_val[1][0]))
plt.plot(processed_datas["square"], processed_datas['price'], 'ro')
caclated_prices = []
for square in processed_datas["square"]:
    caclated_prices.append(final_w_val[0][0] + final_w_val[1][0] * square)
plt.plot(processed_datas["square"], caclated_prices)
plt.show()
from dataset_util import *
from image_util import *
from data_model_util import *

# create_train_dataset('./train_data_sets', 100, 60, 4, 3000)
# create_train_dataset('./test_data_sets', 100, 60, 4, 1000)

train_images, train_labels = read_dataset('./train_data_sets')
test_images, test_labels = read_dataset('./test_data_sets')
# # show_images(images, labels, 4, 4)
#
#
# 准备数据， image shape为 100, 60, 1, label是4个10长度的onehot数据
train_images, shape = normalize_images(train_images)
train_labels = normalize_labels(train_labels)

test_images, shape = normalize_images(test_images)
test_labels = normalize_labels(test_labels)

#
train_model = create_cnn_model(shape)
save_model_flow_as_image(train_model, "./models_vis/cnn_model.png")

#
history = train_model.fit(train_images, train_labels, batch_size=500, epochs=100, verbose=2, validation_data=(test_images, test_labels))
save_history(history, "./history/epoch_100.his")
plot_train_history(history)



# 反向验证 。。。 
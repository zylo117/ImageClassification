import os
import tensorflow as tf
from keras.backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gm_config = tf.ConfigProto()
gm_config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=gm_config))

from keras.applications.densenet import DenseNet121
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
# model = AlexNet.build(width=227, height=227, depth=3, classes=2)
# model = InceptionResNetV2(weights='imagenet',input_shape=(299, 299, 3), include_top=False)
model = DenseNet121(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
plot_model(model, to_file="InceptionResNetV2_notop.png", show_shapes=True)

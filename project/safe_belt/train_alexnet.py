# set the matplotlib backend so figures can be saved in the background
import argparse
import os

from custom_nn.alexnet import AlexNet

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys

sys.path.append('../../')

import matplotlib
from keras import Model, Input
from keras.applications.densenet import DenseNet121
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
from sklearn.metrics import classification_report

from callbacks.epochcheckpoint import EpochCheckpoint
from custom_nn.fcheadnet import FCHeadNet
from project.safe_belt import config
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from preprocessing.patchpreprocessor import PatchPreprocessor
from preprocessing.meanpreprocessor import MeanPreprocessor
from callbacks.trainingmonitor import TrainMonitor
from tools.io_.hdf5datasetgenrator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adamax
import tensorflow as tf
from keras.backend import set_session
from keras.utils import multi_gpu_model
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-pt', '--pre_trained_model', help="path to base model to finetune on",
                default='../../pre_trained_models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

G = 1
if G > 1:
    print("[INFO] setting up for multi-gpu")
    gm_config = tf.ConfigProto()
    gm_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=gm_config))

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
# pp = PatchPreproce?ssor(224, 224)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, aug=aug,
                                preprocessors=[sp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128,
                              preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print('[INFO] compiling model...')
opt = Adamax(lr=0.01)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
single_gpu_model = AlexNet.build(227, 227, 3, 2)

if G <= 1:
    print("[INFO] training with 1 GPU...")
    model = single_gpu_model
    # otherwise, we are compiling using multiple GPUs
else:
    print("[INFO] training with {} GPUs...".format(G))

    # we'll store a copy of the model on *every* GPU and then combine
    # the results from the gradient updates on the CPU
    # make the model parallel
    model = multi_gpu_model(single_gpu_model, gpus=G)

model.compile(loss='binary_crossentropy', optimizer=opt,
              metrics=['accuracy'])

EPOCH = 50
BS = 128

# construct the set of callbacks
lr_schedule = lambda epoch: 0.001 * 0.95 ** epoch
learning_rate = np.array([lr_schedule(i) for i in range(EPOCH)])

callbacks = [
    TrainMonitor('logs/monitor_{}.png'.format(os.getpid()), jsonPath='logs/monitor_{}.json'.format(os.getpid())),
    EarlyStopping(patience=5, monitor='val_acc'),
    EpochCheckpoint('logs/ckpt/', every=1),
    TensorBoard(log_dir='logs'),
    LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // BS,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // BS,
    epochs=EPOCH,
    max_queue_size=BS * G,
    callbacks=callbacks, verbose=1
)

# save the model to file
print('[INFO] serializing model...')
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()

from project.safe_belt import config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.simplepreprocessor import SimplePreprocessor
from tools.io_.hdf5datasetwriter import HDF5DatasetWriter
from tools import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the paths to the images
trainPaths = paths.list_images(config.IMAGE_PATH)
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the
# testing split from the training data
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.RATIO_TEST_IMAGES, stratify=trainLabels,
                         random_state=42)
trainPaths, testPaths, trainLabels, testLabels = split

# perform another stratified sampling, this time to build the
# validation data
split = train_test_split(trainPaths, trainLabels,
                         test_size=config.RATIO_VAL_IMAGES, stratify=trainLabels,
                         random_state=42)
trainPaths, valPaths, trainLabels, valLabels = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
    ('train', trainPaths, trainLabels, config.TRAIN_HDF5),
    ('val', valPaths, valLabels, config.VAL_HDF5),
    ('test', testPaths, testLabels, config.TEST_HDF5),
]

# initialize the image preprocessor and the lists of RGB channel
# averages
# aap = AspectAwarePreprocessor(224, 224)
sp = SimplePreprocessor(224, 224)
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # loop over the dataset tuples
    print('[INFO] building {}...'.format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 224, 224, 3), outputPath, overwrite=True, bufSize=128, compression=0)

    # initialize the progress bar
    widgets = ['Building Dataset: ', progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
                                   widgets=widgets).start()

    # initialize the progress bar
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = sp.preprocess(image)

        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == 'train':
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

    # close the HDF5 writer
    pbar.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()

# define the paths to the images directory
IMAGE_PATH = '../../datasets/safe_belt/images'

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
RATIO_VAL_IMAGES = 0.2
RATIO_TEST_IMAGES = 0.2

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = '../../datasets/safe_belt/hdf5/train.hdf5'
VAL_HDF5 = '../../datasets/safe_belt/hdf5/val.hdf5'
TEST_HDF5 = '../../datasets/safe_belt/hdf5/test.hdf5'

# path to the output model file
MODEL_PATH = 'output/densenet_safe_belt.model'

# path to the output model file
DATASET_MEAN = 'output/mean.json'

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = 'output'

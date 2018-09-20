from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense


class FCHeadNet:
    @staticmethod
    def build(baseModel, classes, D, pooling=False):
        # initialize the head model that will be placed on top of
        # the base, then add a FC layer
        headModel = baseModel.output

        # FC block 1
        if not pooling:
            headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D * 4, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.5)(headModel)

        # FC block 2
        headModel = Dense(D, activation="relu")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        # return the model
        return headModel



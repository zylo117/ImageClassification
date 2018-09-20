from keras.engine.saving import load_model
from sklearn.preprocessing import LabelEncoder

img_paths = '../../datasets/safe_belt/'

lb = LabelEncoder()
testY = lb.transform(testY)

model = load_model('output/densenet_safe_belt_v1.model')
model.predict()

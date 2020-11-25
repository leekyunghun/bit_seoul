import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak                          # shape 맞출 필요가 없다. 다 자동으로 해준다.

x_train = np.load('./data/keras63_train_x.npy') 
y_train = np.load('./data/keras63_train_y.npy')
x_test = np.load('./data/keras63_test_x.npy')
y_test = np.load('./data/keras63_test_y.npy')

print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

clf = ak.ImageClassifier(overwrite = True, max_trials = 1)
clf.fit(x_train, y_train, epochs = 50)

predicted_y = clf.predict(x_test)
print(predicted_y)

print(clf.evaluate(x_test, y_test))


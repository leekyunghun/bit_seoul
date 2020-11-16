from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()        # (28, 28, 1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(x_train[0])
print("=================")
print(y_train[0])

plt.imshow(x_train[0], 'gray')
plt.show()
import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype("float32") / 255.0
x_test = x_test.reshape(10000,28,28,1).astype("float32") / 255.0

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min = 0, a_max = 1)
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max = 1)

from tensorflow.keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model

def autoencoder(filters_size, kernel_size):
    model = Sequential()
    model.add(Conv2D(filters = 784, kernel_size = kernel_size, padding = 'same', input_shape = (28, 28, 1), activation='relu'))
    model.add(Conv2D(filters = filters_size, kernel_size = kernel_size, padding = 'same', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters = filters_size - 256, kernel_size = kernel_size, padding = 'same', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Conv2D(filters = filters_size, kernel_size = kernel_size, padding = 'same', activation='relu'))
    model.add(Conv2D(filters = 1, kernel_size = kernel_size, padding = 'same', activation='sigmoid'))

    return model

model = autoencoder(512, (2,2))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])

model.fit(x_train_noised, x_train_noised, epochs = 10, batch_size=100)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize = (20,7))

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 원본 이미지를 맨위에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("NOISE", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
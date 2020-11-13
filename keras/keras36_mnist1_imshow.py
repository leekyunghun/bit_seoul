import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_test[1000])
print(y_test[1000])

plt.imshow(x_test[1000], 'gray')
plt.show()


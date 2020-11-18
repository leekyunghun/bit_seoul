# 데이터 6개를 저장하시오.
from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
import numpy as np

(cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()

np.save('./data/cifar10_x_train.npy', arr = cifar10_x_train)
np.save('./data/cifar10_y_train.npy', arr = cifar10_y_train)
np.save('./data/cifar10_x_test.npy', arr = cifar10_x_test)
np.save('./data/cifar10_y_test.npy', arr = cifar10_y_test)

(fashion_mnist_x_train, fashion_mnist_y_train), (fashion_mnist_x_test, fashion_mnist_y_test) = fashion_mnist.load_data()
np.save('./data/fashion_mnist_x_train.npy', arr = fashion_mnist_x_train)
np.save('./data/fashion_mnist_y_train.npy', arr = fashion_mnist_y_train)
np.save('./data/fashion_mnist_x_test.npy', arr = fashion_mnist_x_test)
np.save('./data/fashion_mnist_y_test.npy', arr = fashion_mnist_y_test)

(cifar100_x_train, cifar100_y_train), (cifar100_x_test, cifar100_y_test) = cifar100.load_data()

np.save('./data/cifar100_x_train.npy', arr = cifar100_x_train)
np.save('./data/cifar100_y_train.npy', arr = cifar100_y_train)
np.save('./data/cifar100_x_test.npy', arr = cifar100_x_test)
np.save('./data/cifar100_y_test.npy', arr = cifar100_y_test)

boston_dataset = load_boston()
boston_x = boston_dataset.data               
boston_y = boston_dataset.target
np.save('./data/boston_x.npy', arr = boston_x)
np.save('./data/boston_y.npy', arr = boston_y)

diabetes_dataset = load_diabetes()
diabetes_x = diabetes_dataset.data               
diatest_y = diabetes_dataset.target
np.save('./data/diabetes_x.npy', arr = diabetes_x)
np.save('./data/diatest_y.npy', arr = diatest_y)

cancer_dataset = load_breast_cancer()
cancer_x = cancer_dataset.data               
cancer_y = cancer_dataset.target
np.save('./data/cancer_x.npy', arr = cancer_x)
np.save('./data/cancer_y.npy', arr = cancer_y)

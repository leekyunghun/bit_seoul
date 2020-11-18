import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

# print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

# np.save('./data/mnist_x_train.npy', arr = x_train)
# np.save('./data/mnist_x_test.npy', arr = x_test)
# np.save('./data/mnist_y_train.npy', arr = y_train)
# np.save('./data/mnist_y_test.npy', arr = y_test)

x_train = np.load('./data/mnist_x_train.npy')                   # np.save와 똑같이 np.load만 하면된다.
x_test = np.load('./data/mnist_x_test.npy')
y_train = np.load('./data/mnist_y_train.npy')
y_test = np.load('./data/mnist_y_test.npy')

print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255.

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import load_model

#==================CHECKPOINT=======================#
model1 = load_model('./model/mnist/03- 0.060129.hdf5')
result1 = model1.evaluate(x_test, y_test, batch_size = 1)

#==================LOAD_MODEL=======================#
model2 = load_model('./save/mnist/mnist1_model_2.h5')
result2 = model2.evaluate(x_test, y_test, batch_size = 1)

#==================LOAD_WEIGHT=======================#
model = Sequential()
model.add(Conv2D(20, (2,2), input_shape = (28, 28, 1), padding = 'same'))           # (28, 28, 10)      activation의 default값은 relu, LSTM의 activation default값은 tanh
model.add(Conv2D(40, (2,2), padding = 'valid'))                                     # (27, 27, 20)
model.add(Conv2D(10, (3,3)))                                                        # (25, 25, 30)
model.add(Conv2D(20, (2,2), strides = 1))                                           # (12, 12, 40)
model.add(MaxPooling2D(pool_size = 2))                                              # (6, 6, 40)
model.add(Flatten())                                                                # (1440, )
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax'))    

# 3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])    # 분류모델에서의 loss는 categorical_crossentropy를 해준다.
model.load_weights('./save/mnist/mnist1_weights.h5')

# 4.평가, 예측
result3 = model.evaluate(x_test, y_test, batch_size = 1)

print("CheckPoint")
print("loss : ", result1[0])
print("accuracy : ", result1[1])

print("\nLoad_model")
print("loss : ", result2[0])
print("accuracy : ", result2[1])

print("\nLoad_weight")
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# CheckPoint
# loss :  0.05871579051017761
# accuracy :  0.9825999736785889

# Load_model
# loss :  0.05871579051017761
# accuracy :  0.9825999736785889

# Load_weight
# loss :  0.05871579051017761
# accuracy :  0.9825999736785889
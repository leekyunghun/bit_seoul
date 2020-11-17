# Onehotencoding

# 1.keras
# to_categorical() 사용

# 2.sklearn
# OneHotEncoder() 사용

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

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

model = Sequential()
model.add(Conv2D(20, (2,2), input_shape = (28, 28, 1), padding = 'same'))           # (28, 28, 10)      activation의 default값은 relu, LSTM의 activation default값은 tanh
model.add(Conv2D(40, (2,2), padding = 'valid'))                                     # (27, 27, 20)
model.add(Conv2D(10, (3,3)))                                                        # (25, 25, 30)
model.add(Conv2D(20, (2,2), strides = 1))                                           # (12, 12, 40)
model.add(MaxPooling2D(pool_size = 2))                                              # (6, 6, 40)
model.add(Flatten())                                                                # (1440, )
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax'))                                        # 분류 모델은 항상 아웃풋 activation을 softmax해야함

model.summary()

# model.save("./save/model_test01_1.h5")

# 3.컴파일, 훈련
modelpath = "./model/{epoch:02d}-{val_loss: 4f}.hdf5"                               # Checkpoint가 저장될 경로 설정

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')      # Model Checkpoint monitor로 지정한 값이 좋을때마다 저장 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])    # 분류모델에서의 loss는 categorical_crossentropy를 해준다.
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2, verbose = 1, callbacks = [early_stopping, cp])

# 모델 + 가중치 저장
model.save("./save/model_test02_2.h5")

# 가중치 저장
model.save_weights("./save/weight_test02.h5")

# 4.평가, 예측
result = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", result[0])
print("accuracy : ", result[1])

# Epoch 10/10
# 1500/1500 [==============================] - 4s 2ms/step - loss: 0.0095 - accuracy: 0.9970 - val_loss: 0.0882 - val_accuracy: 0.9832
# 313/313 [==============================] - 1s 2ms/step - loss: 0.0873 - accuracy: 0.9837
# loss :  0.08729441463947296
# accuracy :  0.9836999773979187
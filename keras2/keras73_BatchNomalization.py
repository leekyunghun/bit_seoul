# Onehotencoding

# 1.keras
# to_categorical() 사용

# 2.sklearn
# OneHotEncoder() 사용

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255.

# 2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2

model = Sequential()
model.add(Conv2D(20, (2,2), input_shape = (28, 28, 1), padding = 'same'))         
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(40, (2,2), kernel_initializer = 'he_normal'))          # kernel_initializer
model.add(BatchNormalization())                                         # Dropout역할도 해준다
model.add(Activation('relu'))

model.add(Conv2D(10, (3,3), kernel_regularizer = l1(0.01)))             # kernel_regularizer
model.add(Dropout(0.2))

model.add(Conv2D(20, (2,2), strides = 1))                                   
model.add(MaxPooling2D(pool_size = 2))      
                                       
model.add(Flatten())                                                             
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax'))                                      

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard         # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'min')
to_list = TensorBoard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])    # 분류모델에서의 loss는 categorical_crossentropy를 해준다.
model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 100)
print("loss : ", loss)
print("accuracy : ", accuracy)




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
model.add(Conv2D(20, (2,2), input_shape = (28, 28, 1), padding = 'same'))           # (28, 28, 10)      
model.add(Conv2D(40, (2,2), padding = 'valid'))                                     # (27, 27, 20)
model.add(Conv2D(10, (3,3)))                                                        # (25, 25, 30)
model.add(Conv2D(20, (2,2), strides = 1))                                           # (12, 12, 40)
model.add(MaxPooling2D(pool_size = 2))                                              # (6, 6, 40)
model.add(Flatten())                                                                # (1440, )
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax'))                                

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 6, mode = 'min')
ck = ModelCheckpoint("./model", save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)        # patience: n번안에 개선이 없으면 실행, factor: learning rate를 50%로 감소

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])  
model.fit(x_train, y_train, epochs = 30, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping, ck, reduce_lr])

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)
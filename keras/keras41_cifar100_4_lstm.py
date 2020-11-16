from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# 1.데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()      # train set = (?, 32, 32, 3), test set = (? , 1)

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], x_train.shape[3]) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], x_test.shape[3]) / 255.0

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape = (32 * 32, 3))
lstm = LSTM(30)(input) 
dense1 = Dense(100, activation = 'relu')(lstm)
# dense2 = Dense(200, activation = 'relu')(dense1)
# dense3 = Dense(50, activation = 'relu')(dense2)
dense4 = Dense(200, activation = 'relu')(dense1)
output = Dense(100, activation = 'softmax')(dense4)

model = Model(inputs = input, outputs = output)
model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks=[early_stopping])

#4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 100)
print("loss : ", loss)
print("accuracy : ", accuracy)

# accuracy = 
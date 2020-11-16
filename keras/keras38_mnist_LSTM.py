import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2] , 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] , 1) / 255.0

print(x_train.shape, x_test.shape)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape=(28 * 28, 1))
lstm = LSTM(1000, activation = 'relu')(input) 
dense1 = Dense(1000, activation = 'relu')(lstm)
dense2 = Dense(300, activation = 'relu')(dense1)
dense3 = Dense(100, activation = 'relu')(dense2)
output = Dense(10, activation = 'softmax')(dense3)

model = Model(inputs = input, outputs = output)
model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 5, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.2, verbose = 1, callbacks=[early_stopping])

#4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", loss)
print("accuracy : ", accuracy)

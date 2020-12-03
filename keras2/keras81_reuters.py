from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, BatchNormalization, Activation

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000, test_split = 0.2)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (8982,) (2246,) (8982,) (2246,)
print(x_train[0])
print(y_train[0])

x_train = pad_sequences(x_train, maxlen = 1000, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = 1000, padding = 'pre')

print(len(x_train[0]))
print(len(x_train[11]))

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리 : ", category)      # 46

# y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)
print(y_bunpo)
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# 실습 1
# 모델 구성, 완료, 끝
model = Sequential()
model.add(Embedding(10000, 1024, input_length = 1000))
model.add(LSTM(128))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(46, activation='softmax'))   

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard        
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30, batch_size = 32, verbose = 1, callbacks = [early_stopping])
 
accuracy = model.evaluate(x_test, y_test)[1] 
print("accuracy : ", accuracy)

# 71/71 [==============================] - 2s 34ms/step - loss: 2.4239 - accuracy: 0.7280
# accuracy :  0.7279608249664307
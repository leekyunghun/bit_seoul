from tensorflow.keras.datasets import reuters, imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D, BatchNormalization, Activation

# 소스를 완성하시오. Embedding

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (25000,) (25000,) (25000,) (25000,)

x_train = pad_sequences(x_train, padding = 'pre')       # maxlen을 사용하면 해당 값만큼만으로 줄여줌 -> 100개 보다 많은면 100개로 줄어들고 적으면 적은만큼 0으로 padding 해줌
x_test = pad_sequences(x_test, padding = 'pre')

print(len(x_train[0]))
print(len(x_train[11]))

# y의 카테고리 개수 출력
category = np.max(y_train) + 1
print("카테고리 : ", category)      # 2

# y의 유니크한 값을 출력
y_bunpo = np.unique(y_train)       # [0 1]
print(y_bunpo)

model = Sequential()
model.add(Embedding(10000, 1024, input_length = 2494))
model.add(LSTM(64))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(8))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(4))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))   

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard        
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 30, batch_size = 32, verbose = 1, callbacks = [early_stopping])

accuracy = model.evaluate(x_test, y_test)[1] 
print("accuracy : ", accuracy)

# 782/782 [==============================] - 54s 70ms/step - loss: 0.7816 - accuracy: 0.8698
# accuracy :  0.8697599768638611
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2] , 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] , 1) / 255.0

print(x_train.shape, x_test.shape)

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten

model = Sequential()
model.add(Conv1D(64, kernel_size = 2, input_shape = (28 * 28, 1)))
model.add(MaxPooling1D())
model.add(Conv1D(32, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(40, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 3, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks=[early_stopping])

#4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 10/10
# 480/480 [==============================] - 3s 5ms/step - loss: 0.0187 - accuracy: 0.9936 - val_loss: 0.0947 - val_accuracy: 0.9767
# 10000/10000 [==============================] - 15s 1ms/step - loss: 0.0790 - accuracy: 0.9797
# loss :  0.07897961884737015
# accuracy :  0.9797000288963318
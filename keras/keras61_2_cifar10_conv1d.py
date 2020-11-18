from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import Flatten, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(50000, 32 * 32, 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32 * 32, 3).astype("float32") / 255.

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(64, kernel_size = 2, input_shape = (32 * 32, 3)))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(32, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 15, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 50, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks=[early_stopping])

#4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 50/50
# 400/400 [==============================] - 2s 6ms/step - loss: 0.8087 - accuracy: 0.7070 - val_loss: 1.2449 - val_accuracy: 0.5975
# 10000/10000 [==============================] - 13s 1ms/step - loss: 1.2687 - accuracy: 0.5894
# loss :  1.2687201499938965
# accuracy :  0.5893999934196472
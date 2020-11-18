from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import Flatten, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
import numpy as np

# 1.데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()        # (28, 28, 1)

# 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2], 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2], 1) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 2. 모델 구성
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

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 3, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_test, y_test, epochs = 10, batch_size = 100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 10/10
# 80/80 [==============================] - 0s 5ms/step - loss: 0.2700 - accuracy: 0.8986 - val_loss: 0.3772 - val_accuracy: 0.8605
# 10000/10000 [==============================] - 14s 1ms/step - loss: 0.2506 - accuracy: 0.9091
# loss :  0.25062066316604614
# accuracy :  0.9090999960899353
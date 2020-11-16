from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()        

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32 * 32 * 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32 * 32 * 3).astype("float32") / 255.

# 2. 모델 구성
model = Sequential()
model.add(Dense(3000, activation = 'relu', input_shape = (32 * 32 * 3, )))
model.add(Dense(2000, activation = 'relu'))
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min') 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_test, y_test, epochs = 100, batch_size = 32, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", loss)
print("accuracy : ", accuracy)

# # accuracy = 250/250 [==============================] - 1s 4ms/step - loss: 0.3057 - accuracy: 0.9121 - val_loss: 15.9307 - val_accuracy: 0.1230
# 313/313 [==============================] - 0s 2ms/step - loss: 3.4268 - accuracy: 0.7554
# loss :  3.4268414974212646            val_loss: 15.9307
# accuracy :  0.7554000020027161        val_accuracy: 0.1230
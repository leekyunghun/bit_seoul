from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import Flatten, MaxPooling1D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()        

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32 * 32 , 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32 * 32 , 3).astype("float32") / 255.

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv1D(64, kernel_size = 2, input_shape = (32 * 32, 3)))
model.add(MaxPooling1D())
model.add(Conv1D(32, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D())
model.add(Dropout(0.2))
model.add(Conv1D(40, kernel_size = 2, activation = 'relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 3, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_test, y_test, epochs = 10, batch_size=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 10/10
# 80/80 [==============================] - 1s 7ms/step - loss: 2.5511 - accuracy: 0.3445 - val_loss: 3.4085 - val_accuracy: 0.2130
# 10000/10000 [==============================] - 14s 1ms/step - loss: 2.5122 - accuracy: 0.3706
# loss :  2.512197971343994
# accuracy :  0.37059998512268066
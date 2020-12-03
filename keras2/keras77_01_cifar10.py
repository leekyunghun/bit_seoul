from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

x_train = x_train.reshape(50000, 32, 32, 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32, 32, 3).astype("float32") / 255.

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(40, (2,2), input_shape = (32, 32, 3), padding = 'valid'))           # (31, 31, 20)     
model.add(Conv2D(60, (2,2), padding = 'valid'))                                     # (30, 30, 40)
model.add(Conv2D(30, (3,3)))                                                        # (28, 28, 10)
model.add(Conv2D(10, (2,2), strides = 1))                                           # (27, 27, 20)
model.add(MaxPooling2D(pool_size = 2))                                              # (13, 13, 20)
model.add(Flatten())                                                                # (3380, )
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax')) 

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

# 313/313 [==============================] - 1s 2ms/step - loss: 4.2994 - accuracy: 0.5885
# loss :  4.299354076385498
# accuracy :  0.5885000228881836
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

x_train = x_train.reshape(50000, 32 , 32 , 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32 , 32 , 3).astype("float32") / 255.

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2,2), activation = 'relu', input_shape = (32, 32, 3), padding = 'valid'))           # (31, 31, 20)     
model.add(Conv2D(600, (2,2), activation = 'relu', padding = 'valid'))                                     # (30, 30, 40)
model.add(Conv2D(300, (3,3), activation = 'relu'))                                                        # (28, 28, 10)
model.add(Conv2D(100, (2,2), activation = 'relu', strides = 1))                                           # (27, 27, 20)
model.add(MaxPooling2D(pool_size = 2))                                              # (13, 13, 20)
model.add(Flatten())                                                                # (3380, )
model.add(Dense(1000, activation = 'relu'))                                          
model.add(Dense(500, activation = 'relu'))                                          
model.add(Dense(100, activation = 'softmax')) 

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_test, y_test, epochs = 100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", loss)
print("accuracy : ", accuracy)

# accuracy = 250/250 [==============================] - 7s 29ms/step - loss: 0.0350 - accuracy: 0.9929 - val_loss: 12.6707 - val_accuracy: 0.1635
# 313/313 [==============================] - 3s 8ms/step - loss: 2.5679 - accuracy: 0.8237
# loss :  2.567854404449463         val_loss: 12.6707
# accuracy :  0.8237000107765198    val_accuracy: 0.1635
from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Dropout
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
model.add(Dropout(0.2))
model.add(Conv2D(600, (2,2), activation = 'relu', padding = 'valid'))                                     # (30, 30, 40)
model.add(Conv2D(300, (3,3), activation = 'relu'))                                                        # (28, 28, 10)
model.add(Dropout(0.2))
model.add(Conv2D(100, (2,2), activation = 'relu', strides = 1))                                           # (27, 27, 20)
model.add(MaxPooling2D(pool_size = 2))                                              # (13, 13, 20)
model.add(Dropout(0.2))
model.add(Flatten())                                                                # (3380, )
model.add(Dense(1000, activation = 'relu'))                                          
model.add(Dropout(0.2))
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

predict = model.predict(x_test)
pred = [np.argmax(predict[i]) for i in range(9400, 9420)]
print(pred)

y_test_recovery = np.argmax(y_test, axis=1).reshape(-1,1)                   # reshape(-1, 1)은 열 갯수에 맞춰서 행을 자동으로 맞춰줌
y_test_recovery = y_test_recovery.reshape(y_test_recovery.shape[1], y_test_recovery.shape[0])
print(y_test_recovery.shape)
print("y_test : ", y_test_recovery[0, 9400:9420])

# accuracy
# 250/250 [==============================] - 8s 31ms/step - loss: 4.6047 - accuracy: 0.0103 - val_loss: 4.6162 - val_accuracy: 0.0060
# 313/313 [==============================] - 3s 8ms/step - loss: 4.6064 - accuracy: 0.0100
# loss :  4.606407642364502
# accuracy :  0.009999999776482582
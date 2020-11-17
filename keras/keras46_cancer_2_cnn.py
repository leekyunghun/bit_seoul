import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Input, MaxPooling2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
dataset = load_breast_cancer()
x = dataset.data                # (569, 30)
y = dataset.target              # (569, )

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

x_train = x_train.reshape(x_train.shape[0], 5, 6, 1)
x_test = x_test.reshape(x_test.shape[0], 5, 6, 1)

# 2.모델 구성
model = Sequential()
model.add(Conv2D(50, (2, 3), input_shape = (5, 6, 1), activation='relu'))       # (4, 4, 50)
model.add(Dropout(0.2))
model.add(Conv2D(30, (2, 3), activation='relu'))       # (3, 2, 30)
model.add(Conv2D(40, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4.예측, 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 100/100
# 30/30 [==============================] - 0s 3ms/step - loss: 0.0188 - accuracy: 0.9933 - val_loss: 0.0468 - val_accuracy: 0.9800
# 6/6 [==============================] - 0s 6ms/step - loss: 0.1262 - accuracy: 0.9708
# loss :  0.12615619599819183
# accuracy :  0.9707602262496948
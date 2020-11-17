# 다중분류
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Input
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
dataset = load_iris()
x = dataset.data                # (150, 4)
y = dataset.target              # (150, )

# 데이터 전처리 1.OneHotEncoding
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
model = Sequential()
model.add(Dense(30, activation = 'relu', input_shape = (4, )))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.25)#, callbacks = [early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("acc: ", acc)

# Epoch 100/100
# 8/8 [==============================] - 0s 3ms/step - loss: 0.0145 - acc: 0.9615 - val_loss: 0.0196 - val_acc: 0.9630
# 45/45 [==============================] - 0s 1ms/step - loss: 0.0134 - acc: 0.9778
# loss :  0.013407221995294094
# acc:  0.9777777791023254
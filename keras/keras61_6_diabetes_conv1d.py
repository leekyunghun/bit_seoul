from sklearn.datasets import load_diabetes              # sklearn에서 dataset 가져오는방법
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np

# 1. 데이터
dataset = load_diabetes()       # sklearn에서 제공되는 dataset load하는 방법
x = dataset.data                # (442, 10)
y = dataset.target              # (442, )

print(x.shape, y.shape)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler       # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)   

from sklearn.model_selection import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1) / 255.
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2.모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape = (10, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(20, 2, activation = 'relu'))
model.add(Conv1D(40, 1, activation = 'relu'))
model.add(Flatten())
model.add(Dense(80, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.2, callbacks = [early_stopping])

# 4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mae : ", mae)

predict = model.predict(x_test)
print("predict : ", predict)

# Epoch 100/100
# 25/25 [==============================] - 0s 3ms/step - loss: 3276.8083 - mae: 46.8927 - val_loss: 3144.8354 - val_mae: 46.1067
# 133/133 [==============================] - 0s 1ms/step - loss: 3144.9861 - mae: 44.0625
# loss :  3144.986083984375
# mae :  44.062469482421875
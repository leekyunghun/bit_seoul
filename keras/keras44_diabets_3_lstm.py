from sklearn.datasets import load_diabetes              # sklearn에서 dataset 가져오는방법
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) / 255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2.모델 구성
model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (10, 1)))
model.add(Dense(512))
model.add(Dropout(0.4))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(128))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse: ", mse)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2) 

# 8/8 [==============================] - 0s 15ms/step - loss: 4815.5679 - mae: 57.1673 - val_loss: 4562.2827 - val_mae: 55.4704
# 133/133 [==============================] - 0s 2ms/step - loss: 4535.9297 - mae: 57.6233
# loss :  4535.9296875
# mse:  57.6232795715332
# RMSE :  67.34930537722835
# R2 :  0.27962199149844214
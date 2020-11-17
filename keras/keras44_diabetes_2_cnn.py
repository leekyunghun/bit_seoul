from sklearn.datasets import load_diabetes              # sklearn에서 dataset 가져오는방법
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1) / 255.
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2.모델 생성
model = Sequential()
model.add(Conv2D(100, (2,1), activation = 'relu', input_shape = (10, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(50, (2,1), activation = 'relu'))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(200))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4.평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse: ", mse)

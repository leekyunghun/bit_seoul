from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model
import numpy as np

# 1. 데이터
dataset = load_boston()
x = dataset.data                # (506, 13)
y = dataset.target              # (506, )

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2. 모델 구성
input = Input(shape = (13, ))
dense1 = Dense(64, activation='relu')(input)
dense1 = Dropout(0.2)(dense1)
dense2 = Dense(64, activation='relu')(dense1)
dense3 = Dense(10, activation='relu')(dense2)
output = Dense(1)(dense3)

model = Model(inputs = input, outputs = output)

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 200, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 500, batch_size = 50, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse: ", mse)


# 1.데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11], [10,11,12],
              [20,30,40], [30,40,50], [40,50,60]])  # (13,3)
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])    # (13, )

x_input = np.array([50,60,70])                      #(3, )

x = x.reshape(13, 3, 1)                             # x를 (13, 3, 1)로 바꿔줌 1개씩 작업하기위해
x_input = x_input.reshape(1, 3, 1)                  # x_input도 x와 똑같이 양식을 맞춰줌

print(x.shape)
print(x_input.shape)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# model = Sequential()
# model.add(LSTM(30, activation = 'relu', input_shape = (3, 1)))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))

input = Input(shape=(3,1))
lstm = LSTM(30, activation = 'relu')(input) 
dense1 = Dense(200)(lstm)
dense2 = Dense(100)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(30)(dense3)
dense5 = Dense(10)(dense4)
output = Dense(1)(dense5)

model = Model(inputs = input, outputs = output)

model.summary()

# 3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping        # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'loss', patience = 100, mode = 'min')        # patience => mode로 원하는 시점에서 ?회까지 더 해보겠다.
# early_stopping = EarlyStopping(monitor = 'loss', patience = 150, mode = 'auto')        # patience => mode로 원하는 시점에서 ?회까지 더 해보겠다.

model.fit(x, y, epochs = 10000, batch_size = 1, callbacks = [early_stopping])         # callbacks = ? 호출할 내용을 넣으면 된다 

# 4.평가, 예측
predict = model.predict(x_input)
print("predict : ", predict)
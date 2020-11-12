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

# 2.모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(30, activation='relu', input_length = 3, input_dim = 1, return_sequences = True))   # input_shape를 input_length, input_dim으로 나눠서 선언가능
model.add(LSTM(30))                     # LSTM 레이어가 2개이상 or 1개만 일때 뭐가 좋은지는 데이터마다 다르므로 이것저것 다 해봐서 알아보는게 좋음
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping        # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'loss', patience = 50, mode = 'min')
model.fit(x, y, epochs = 200, batch_size = 1, callbacks=[early_stopping])

# 4.평가, 예측
predict = model.predict(x_input)
print("predict : ", predict)
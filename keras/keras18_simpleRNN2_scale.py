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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

model = Sequential()
model.add(SimpleRNN(30, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 200, batch_size = 1)

# 4.평가, 예측
predict = model.predict(x_input)
print("predict : ", predict)


#                   predict 값          loss 값             parameter 수
# LSTM               80.36166           0.1813                 4,513
# SimpleRNN          80.04076           0.0671                 2,661
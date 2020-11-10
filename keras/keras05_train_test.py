import numpy as np

#1. 데이터                                          # 선형회귀 예제  
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x_pred = np.array([11,12,13])

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense

#2.모델 구성
model = Sequential()                    # model이 Sequential이라고 선언
model.add(Dense(10, input_dim = 1))    # input_dim = 1 => input이 1차원
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'acc'])  # metrics = 평가방식 (훈련중에 연산한 내용을 보여주는역할)

model.fit(x, y, epochs = 100, batch_size = 1)   # model.fit => 모델을 훈련시킴
# model.fit(x, y, epochs = 100)   # model.fit => 모델을 훈련시킴

#4.평가, 예측
# loss, acc = model.evaluate(x, y, batch_size = 1)
loss = model.evaluate(x, y)                                 # evaluate의 디폴트는 loss값 metrics에 추가한 값들이 evaluate 출력에 포함

print("loss : ", loss)
# print("acc : ", acc)

y_pred = model.predict(x_pred)                   # 예측값 확인
print("결과물: \n :", y_pred)
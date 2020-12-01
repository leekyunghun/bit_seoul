import numpy as np

#1. 데이터                                          # 선형회귀 예제  
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense

#2.모델 구성
model = Sequential()                    # model이 Sequential이라고 선언
model.add(Dense(50, input_dim = 1))    # input_dim = 1 => input이 1차원
model.add(Dense(30))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = Adam(lr = 0.001)
# optimizer = Adadelta(lr = 0.1)
# optimizer = Adamax(lr = 0.002)
# optimizer = Adagrad(lr = 0.003)
# optimizer = RMSprop(lr = 0.001)
# optimizer = SGD(lr = 0.002)
# optimizer = Nadam(lr = 0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])  # metrics = 평가방식
model.fit(x, y, epochs = 100, batch_size = 1)   # model.fit => 모델을 훈련시킴

#4.평가, 예측
loss, mse = model.evaluate(x, y, batch_size = 1)

y_pred = model.predict([11])
print("loss : ", loss, " 결과물 : ", y_pred)

# learning_rate = 0.001
# Adam
# loss :  1.89288580469682e-12  결과물 :  [[10.999997]]

# Adadelta
# loss :  5.816532611846924  결과물 :  [[6.7119846]]

# Adamax
# loss :  0.009117906913161278  결과물 :  [[10.889803]]

# Adagrad
# loss :  0.022785861045122147  결과물 :  [[10.8079605]]

# RMSprop
# loss :  0.24149934947490692  결과물 :  [[10.132137]]

# SGD
# loss :  4.631166575563839e-06  결과물 :  [[10.997268]]

# Nadam
# loss :  1.6768808834988908e-13  결과물 :  [[11.]]
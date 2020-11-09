import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense

#2.모델 구성
model = Sequential()                    # model이 Sequential이라고 선언
model.add(Dense(1, input_dim = 1, activation='relu'))    # input_dim = 1 => input이 1차원
# model.add(Dense(3))
# model.add(Dense(1, activation='relu'))

model.summary()

#3.컴파일, 훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # metrics = 평가방식
model.fit(x, y, epochs = 500, batch_size = 1)   # model.fit => 모델을 훈련시킴
y_predict = model.predict(x)

#4.평가, 예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("acc : ", acc)
print(y_predict)

# 1. 데이터
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])      # (4,3)
y = np.array([4,5,6,7])                                 # (4, )

print("x.shape : ", x.shape)
print("y.shape : ", y.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)                # LSTM은 (행, 열, 몇개씩 잘라서 작업할지의 갯수) 이 형태를 input으로 받기때문에 shape를 맞춰줘야한다.
# x = x.reshape(4, 3, 1)        # 이 방식도 가능

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(30, activation = 'relu', input_shape = (3, 1)))  # input_shape이 (3, 1)인 이유는 x의 [1,2,3]이 y의 4를 만드는 과정에서 1,2,3 3개를 1개씩 순차적으로 학습을 시켜준다는 뜻이다. 
model.add(LSTM(30, activation = 'relu', input_length = 3, input_dim = 1 ))   # input_shape를 input_length, input_dim으로 나눠서 선언가능
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x, y, epochs = 100, batch_size = 1, verbose = 1)

x_input = np.array([5,6,7])
x_input = x_input.reshape(1, 3, 1)

# 4. 예측, 평가
result = model.predict(x_input)
print(result)


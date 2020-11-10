#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711,811), range(100)])
y = np.array(range(101, 201))

# x와 y의 열값이 3개이므로 y1, y2, y3 = (w1 * x1) + (w2 * x2) + (w3 * x3) + b

x = x.T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

#2. 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# model = Sequential()
# model.add(Dense(5, input_shape = (3, ), activation = 'relu'))         # input_dim 값은 열의 갯수
# # (100,10,3) => input_shape = (10,3)
# model.add(Dense(4, activation = 'relu'))
# model.add(Dense(3, activation = 'relu'))
# model.add(Dense(1))

input1 = Input(shape = (3, ))
dense1 = Dense(5, activation = 'relu')(input1)
dense2 = Dense(4, activation = 'relu')(dense1)
dense3 = Dense(3, activation = 'relu')(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs = output1)

model.summary()     # param = weight갯수 + bias갯수

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split = 0.2, verbose = 1)

#4.평가, 예측
loss = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2) 

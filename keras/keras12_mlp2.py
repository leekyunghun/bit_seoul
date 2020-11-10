#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711,811), range(100)])
y = np.array((range(101, 201)))

# x와 y의 열값이 3개이므로 x1, x2, x3 & y1, y2, y3

x = x.T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 3))         # input_dim 값은 열의 갯수
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

#4.평가, 예측
loss = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)

y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2) 

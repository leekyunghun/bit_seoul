#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

x_train = x[:60]        # train 60개
y_train = y[:60]
x_val = x[60:80]        # validation 20개
y_val = y[60:80]
x_test = x[80:]         # test 20개
y_test = y[80:]

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(70, input_dim = 1))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가,예측
loss = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2: ", r2)


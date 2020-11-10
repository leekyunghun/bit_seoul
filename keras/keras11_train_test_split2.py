#1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(101,201))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)      # 주어진 데이터전체에서 train, test set을 자동으로 만들어줌
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.75)         # train, test set 말고 validation set까지 만들고싶으면 train_test_split을 두번하면된다.

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(70, input_dim = 1))
model.add(Dense(70, input_shape = (1, )))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data = (x_val, y_val))

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

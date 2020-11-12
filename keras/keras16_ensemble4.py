#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711,811), range(100)])

y1 = np.array([range(101, 201), range(311,411), range(100)])
y2 = np.array([range(501, 601), range(431,531), range(100, 200)])
y3 = np.array([range(501, 601), range(431,531), range(100, 200)])

x1 = x1.T
y1 = y1.T
y2 = y2.T
y3 = y3.T

from sklearn.model_selection import train_test_split
x1_train, x1_test= train_test_split(x1, train_size = 0.7)
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(y1, y2, y3, train_size = 0.7)

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape = (3, ))
dense1 = Dense(16, activation = 'relu', name = 'model1_1')(input1)
dense2 = Dense(8, activation = 'relu', name = 'model1_2')(dense1)
dense3 = Dense(4, activation = 'relu', name = 'mode1_3')(dense2)
output = Dense(1, name = 'model1_output')(dense3)

######### output 모델 구성 (분기)
output1 = Dense(30, name = 'ouput1_1')(output)
output1 = Dense(7, name = 'output1_2')(output1)
output1 = Dense(3, name = 'output1_3')(output1)

output2 = Dense(15, name = 'output2_1')(output)
output2_1 = Dense(14, name = 'output2_2')(output2)
output2_2 = Dense(11, name = 'output2_3')(output2_1)
output2_3 = Dense(3, name = 'output2_4')(output2_2)

output3 = Dense(30, name = 'ouput3_1')(output)
output3 = Dense(7, name = 'output3_2')(output3)
output3 = Dense(3, name = 'output3_3')(output3)

# 모델 정의
model = Model(inputs = input1, outputs = [output1, output2_3, output3])
# model.summary()

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x1_train, [y1_train, y2_train, y3_train], epochs = 100, batch_size = 8, validation_split = 0.25, verbose = 1)

#4. 예측, 평가
result = model.evaluate(x1_test, [y1_test, y2_test, y3_test], batch_size = 8)
print("result : ", result)

y1_pred, y2_pred, y3_pred = model.predict(x1_test)
print(y1_pred, y2_pred, y3_pred)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

y1_rmse = RMSE(y1_test, y1_pred)
y2_rmse = RMSE(y2_test, y2_pred)
y3_rmse = RMSE(y3_test, y3_pred)
rmse_average = (y1_rmse + y2_rmse + y3_rmse) / 3

print("RMSE : ", rmse_average)

from sklearn.metrics import r2_score

r2_y1 = r2_score(y1_test, y1_pred)
r2_y2 = r2_score(y2_test, y2_pred)
r2_y3 = r2_score(y3_test, y3_pred)
r2_average = (r2_y1 + r2_y2 + r2_y3) / 3

print("R2 : ", r2_average) 

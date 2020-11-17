from sklearn.datasets import load_diabetes              # sklearn에서 dataset 가져오는방법
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential, Model
import numpy as np

# 1. 데이터
dataset = load_diabetes()       # sklearn에서 제공되는 dataset load하는 방법
x = dataset.data                # (442, 10)
y = dataset.target              # (442, )

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler       # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)   

from sklearn.model_selection import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

# 2. 모델 구성
input = Input(shape = (10, ))
dense1 = Dense(64, activation='relu')(input)
dense1 = Dropout(0.5)(dense1)
dense2 = Dense(128)(dense1)
dense3 = Dense(256, activation='relu')(dense2)
dense4 = Dense(128)(dense3)
dense4 = Dropout(0.3)(dense4)
dense5 = Dense(32, activation='relu')(dense4)
output = Dense(1)(dense3)

model = Model(inputs = input, outputs = output)

# 3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

model1 = load_model("./model/diabetes/101- 2887.792969.hdf5")
model2 = load_model("./save/diabetes/diabetes_model_2.h5")
model.load_weights("./save/diabetes/diabetes_weights.h5")

# 4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size = 1)
result2 = model2.evaluate(x_test, y_test, batch_size = 1)
result3 = model.evaluate(x_test, y_test, batch_size = 1)

predict1 = model1.predict(x_test)
predict2 = model2.predict(x_test)
predict3 = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE1 : ", RMSE(y_test, predict1))
print("RMSE2 : ", RMSE(y_test, predict2))
print("RMSE3 : ", RMSE(y_test, predict3))

from sklearn.metrics import r2_score
r2_1 = r2_score(y_test, predict1)
r2_2 = r2_score(y_test, predict2)
r2_3 = r2_score(y_test, predict3)

print("\nCheckPoint")
print("loss : ", result1[0])
print("mse: ", result1[1])
print("R2 : ", r2_1) 

print("\nLoad_model")
print("loss : ", result2[0])
print("mse: ", result2[1])
print("R2 : ", r2_2) 

print("\nLoad_weight")
print("loss : ", result3[0])
print("mse: ", result3[1])
print("R2 : ", r2_3) 

# RMSE1 :  53.886381117669586
# RMSE2 :  53.886381117669586
# RMSE3 :  53.886381117669586

# CheckPoint
# loss :  2903.7421875
# mse:  2903.7421875
# R2 :  0.5585639591524618

# Load_model
# loss :  2903.7421875
# mse:  2903.7421875
# R2 :  0.5585639591524618

# Load_weight
# loss :  2903.7421875
# mse:  2903.7421875
# R2 :  0.5585639591524618
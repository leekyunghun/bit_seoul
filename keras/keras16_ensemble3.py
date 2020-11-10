#1. 데이터
import numpy as np

x1 = np.array([range(1, 101), range(711,811), range(100)])
x2 = np.array([range(4, 104), range(761,861), range(100)])

y1 = np.array([range(101, 201), range(311,411), range(100)])

x1 = x1.T
x2 = x2.T
y1 = y1.T

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test= train_test_split(x1, x2, y1, train_size = 0.7)

#2 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 모델 1
input1 = Input(shape = (3, ))
dense1 = Dense(128, activation = 'relu', name = 'model1_1')(input1)
dense2 = Dense(64, activation = 'relu', name = 'model1_2')(dense1)
dense3 = Dense(16, activation = 'relu', name = 'mode1_3')(dense2)
output1 = Dense(4, name = 'model1_output')(dense3)

# 모델 2
input2 = Input(shape = (3, ))
dense1_2 = Dense(64, activation = 'relu', name = 'model2_1')(input2)
dense2_2 = Dense(32, activation = 'relu', name = 'model2_2')(dense1_2)
dense3_2 = Dense(8, activation = 'relu', name = 'model2_3')(dense2_2)
output2 = Dense(4, name = 'model2_output')(dense3_2)

# 모델 병합, concatenate
from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output2])    # 2개이상을 묶을때는 list로 묶기
merge1 = Concatenate()([output1, output2])

middle1 = Dense(30, activation = 'relu', name = 'middle1_1')(merge1)
middle1 = Dense(7, activation = 'relu', name = 'middle1_2')(middle1)
middle1 = Dense(11, activation = 'relu', name = 'middle1_3')(middle1)

######### output 모델 구성 (분기)
output1 = Dense(30, activation = 'relu', name = 'ouput1_1')(middle1)
output1 = Dense(7, activation = 'relu', name = 'output1_2')(output1)
output1 = Dense(3, name = 'output1_3')(output1)


# 모델 정의
model = Model(inputs = [input1, input2], outputs = output1)
# model.summary()

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x1_train, x2_train], y1_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 1)

#4. 예측, 평가
result = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)
print("result : ", result)

y_pred = model.predict([x1_test, x2_test])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y1_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y_pred)
print("R2 : ", r2) 

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

print(x.shape, y.shape)


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler       # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler = StandardScaler()

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

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 200, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 500, batch_size = 32, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse: ", mse)

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2) 

# 8/8 [==============================] - 0s 3ms/step - loss: 2286.6821 - mae: 37.2734 - val_loss: 3512.4312 - val_mae: 46.4771
# 133/133 [==============================] - 0s 988us/step - loss: 3133.9414 - mae: 44.8588
# loss :  3133.94140625
# mse:  44.85879898071289
# RMSE :  55.98161671169436
# R2 :  0.4478497357998462
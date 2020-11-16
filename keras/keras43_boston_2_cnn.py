from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np

# 1. 데이터
dataset = load_boston()
x = dataset.data                # (506, 13)
y = dataset.target              # (506, )

# 데이터 전처리 1.OneHotEncoding
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1) / 255.

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2.모델 생성
model = Sequential()
model.add(Conv2D(20, (2,1), activation = 'relu', input_shape = (13, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (2,1), activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

model.summary()
 
# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 200, batch_size = 10, verbose = 1, validation_split = 0.25 ,callbacks = [early_stopping])

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

# 27/27 [==============================] - 0s 2ms/step - loss: 19.3056 - mae: 2.9481 - val_loss: 15.6649 - val_mae: 2.2747
# 152/152 [==============================] - 0s 2ms/step - loss: 19.5484 - mae: 2.8303
# loss :  19.548437118530273
# mse:  2.8302736282348633
# RMSE :  4.421361360518395
# R2 :  0.7756312608978917
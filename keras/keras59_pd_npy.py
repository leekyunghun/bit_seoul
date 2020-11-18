import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout,Flatten

datasets = pd.read_csv("./data/csv/iris_ys2.csv", header = None, index_col = None, sep = ',')
print(datasets)

aaa = datasets.to_numpy()

pd_aaa = pd.DataFrame(aaa)
pd_aaa.to_csv('./data/csv/iris_ys2_pd.csv', index = False)           # numpy를 csv로 저장하는법

x = aaa[:, :-1]
y = aaa[:, -1]

print(x.shape)
print(y.shape)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)     # sklearn 사용할때 데이터 나누기 아주좋음

x_train = x_train.reshape(x_train.shape[0], 2, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 2, 2, 1)

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
model = Sequential()
model.add(Conv2D(20, (2, 2), input_shape = (2, 2, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (1, 1), input_shape = (2, 2, 1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

# 3.컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.load_weights("./save/iris/iris_weights.h5")

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)

print("loss : ", loss)
print("accuracy : ", accuracy)

predict = model.predict(x_test)

predict = model.predict(x_test)
pred = [np.argmax(predict[i]) for i in range(10, 20)]
print(pred)

y_test_recovery = np.argmax(y_test, axis=1).reshape(-1,1)                   # reshape(-1, 1)은 열 갯수에 맞춰서 행을 자동으로 맞춰줌
y_test_recovery = y_test_recovery.reshape(y_test_recovery.shape[1], y_test_recovery.shape[0])
print(y_test_recovery.shape)
print("y_test : ", y_test_recovery[0, 10:20])
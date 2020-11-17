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
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)   

from sklearn.model_selection import train_test_split
x_train,  x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)

# 2. 모델 구성
model=Sequential()
model.add(Dense(128, activation='relu', input_shape=(10, )))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련
modelpath = "./model/diabetes/{epoch:02d}-{val_loss: 4f}-third.hdf5"                               # Checkpoint가 저장될 경로 설정

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')      # Model Checkpoint monitor로 지정한 값이 좋을때마다 저장 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 1, validation_split = 0.25, callbacks = [early_stopping, cp])

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mse: ", mse)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
mse = hist.history['mse']
val_mse = hist.history['val_mse']

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))       # 단위가 무엇인지 찾아보기
plt.subplot(2, 1, 1)                # 2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)                # 2행 1열 중 첫번째
plt.plot(hist.history['mse'], marker = '.', c = 'red')
plt.plot(hist.history['val_mse'], marker = '.', c = 'blue')
plt.grid()

plt.title('mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['mse', 'val_mse'])

plt.show()
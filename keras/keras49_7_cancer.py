import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Input, MaxPooling2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
dataset = load_breast_cancer()
x = dataset.data                # (569, 30)
y = dataset.target              # (569, )

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

# 2.모델 구성
model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (30, )))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3.컴파일, 훈련
modelpath = "./model/cancer/{epoch:02d}-{val_loss: 4f}.hdf5"                               # Checkpoint가 저장될 경로 설정

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')      # Model Checkpoint monitor로 지정한 값이 좋을때마다 저장 

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.25, callbacks = [early_stopping, cp])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy: ", accuracy)

pred = model.predict(x_test)
print(np.round(pred))
print(y_test)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

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
plt.plot(hist.history['accuracy'], marker = '.', c = 'red')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()
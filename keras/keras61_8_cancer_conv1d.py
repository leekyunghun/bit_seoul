import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM, Input, MaxPooling1D, Flatten
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

x_train = x_train.reshape(x_train.shape[0], 30, 1)
x_test = x_test.reshape(x_test.shape[0], 30, 1)

# 2.모델 구성
model = Sequential()
model.add(Conv1D(32, 2, input_shape = (30, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(64, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy: ", accuracy)

pred = model.predict(x_test)
pred = np.round(pred)
pred = pred.T
print(pred)
print(y_test)

# Epoch 100/100
# 32/32 [==============================] - 0s 2ms/step - loss: 0.0070 - accuracy: 1.0000 - val_loss: 0.0878 - val_accuracy: 0.9750
# 171/171 [==============================] - 0s 1ms/step - loss: 0.1231 - accuracy: 0.9649
# loss :  0.12307287752628326
# accuracy:  0.9649122953414917
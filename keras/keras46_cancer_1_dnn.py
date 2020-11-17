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
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min') 

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("accuracy: ", accuracy)

pred = model.predict(x_test)
print(np.round(pred))
print(y_test)

# Epoch 24/50
# 30/30 [==============================] - 0s 2ms/step - loss: 0.1034 - acc: 0.9564 - val_loss: 0.1144 - val_acc: 0.9700
# 171/171 [==============================] - 0s 1ms/step - loss: 0.1007 - acc: 0.9708
# loss :  0.10068424046039581
# acc:  0.9707602262496948

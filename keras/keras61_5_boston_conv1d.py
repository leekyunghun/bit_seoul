from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np

# 1. 데이터
dataset = load_boston()
x = dataset.data                # (506, 13)
y = dataset.target              # (506, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1) / 255.
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1) / 255.
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 2.모델 구성
model = Sequential()
model.add(Conv1D(64, 3, input_shape = (13, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(30, 2, activation = 'relu'))
model.add(Conv1D(20, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.2, callbacks = [early_stopping])

# 4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mae : ", mae)

predict = model.predict(x_test)
print("predict : ", predict)

# Epoch 100/100
# 29/29 [==============================] - 0s 2ms/step - loss: 60.6136 - mae: 5.4360 - val_loss: 64.0979 - val_mae: 4.8204
# 152/152 [==============================] - 0s 1ms/step - loss: 66.2321 - mae: 5.6299
# loss :  66.2320785522461
# mae :  5.629907131195068
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
dataset = load_iris()
x = dataset.data                # (150, 4)
y = dataset.target              # (150, )

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

x_train = x_train.reshape(x_train.shape[0], 2 * 2, 1)
x_test = x_test.reshape(x_test.shape[0], 2 * 2, 1)

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
model = Sequential()
model.add(Conv1D(20, 2, input_shape = (2 * 2, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(30, 1, activation = 'relu', input_shape = (2 * 2, 1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 10, verbose = 1, validation_split = 0.25, callbacks = [early_stopping])

# 4.예측, 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

# Epoch 100/100
# 8/8 [==============================] - 0s 3ms/step - loss: 0.1665 - accuracy: 0.9103 - val_loss: 0.0907 - val_accuracy: 0.9630
# 2/2 [==============================] - 0s 8ms/step - loss: 0.0808 - accuracy: 0.9556
# loss :  0.08079870790243149
# accuracy :  0.9555555582046509
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

wine = pd.read_csv("./data/csv/winequality-white.csv", header = 0, index_col = None, sep = ';')

wine = wine.values

# 1. 데이터
x = wine[:, :-1]
y = wine[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

from tensorflow.keras.utils import to_categorical              
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(1024, input_shape = (11, )))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint             
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 500)

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs= 500, batch_size= 100, verbose= 1, validation_split= 0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

y_predict = model.predict(x_test)
y_test_recovery = np.argmax(y_test, axis=1).reshape(-1,1)                   # reshape(-1, 1)은 열 갯수에 맞춰서 행을 자동으로 맞춰줌
y_test_recovery = y_test_recovery.reshape(y_test_recovery.shape[1], y_test_recovery.shape[0])
print(y_test_recovery)
# 실습 pca로 축소해서 모델을 완성
# 1. 축소한 차원의 가치가 0.95 이상    -> 2
# 2. 축소한 차원의 가치가 1.0 이상     -> 13

import numpy as np
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_boston()
x = dataset.data                # (506, 13)
y = dataset.target              # (506, )

pca1 = PCA(n_components=2)
pca1 = pca1.fit_transform(x)

pca2 = PCA(n_components=13)
pca2 = pca2.fit_transform(x)

pca1_x_train, pca1_x_test, y_train, y_test = train_test_split(pca1, y, train_size = 0.8, shuffle = True, random_state = 66)
pca2_x_train, pca2_x_test, y_train, y_test = train_test_split(pca2, y, train_size = 0.8, shuffle = True, random_state = 66)

# 2. 모델 구성
model1 = Sequential()
model1.add(Dense(50, activation='relu', input_shape=(2,)))
model1.add(Dense(40, activation='relu'))
model1.add(Dropout(0.1))
model1.add(Dense(30, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(20, activation='relu'))
model1.add(Dense(10, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(5, activation='relu'))
model1.add(Dense(1))

model2 = Sequential()
model2.add(Dense(50, activation='relu', input_shape=(13,)))
model2.add(Dense(40, activation='relu'))
model2.add(Dropout(0.1))
model2.add(Dense(30, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(20, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(5, activation='relu'))
model2.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') 

model1.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model1.fit(pca1_x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

model2.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model2.fit(pca2_x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4. 평가, 예측
loss1, mse1 = model1.evaluate(pca1_x_test, y_test, batch_size = 1)
loss2, mse2 = model2.evaluate(pca2_x_test, y_test, batch_size = 1)

y_predict_1 = model1.predict(pca1_x_test)
y_predict_2 = model2.predict(pca2_x_test)

from sklearn.metrics import r2_score
r2_1 = r2_score(y_test, y_predict_1)
r2_2 = r2_score(y_test, y_predict_2)

print("loss1 : ", loss1)
print("mse1 : ", mse1)
print("r2 : ", r2_1)
print("loss2 : ", loss2)
print("mse2 : ", mse2)
print("r2 : ", r2_2)


# loss1 :  135.99696350097656
# mse1 :  8.880231857299805
# r2 :  -0.6270897122828458
# loss2 :  55.00022506713867
# mse2 :  5.371317386627197
# r2 :  0.3419683633325864
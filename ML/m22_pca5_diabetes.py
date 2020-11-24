# 실습 pca로 축소해서 모델을 완성
# 1. 축소한 차원의 가치가 0.95 이상    -> 8
# 2. 축소한 차원의 가치가 1.0 이상     -> 10

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split

# 1. 데이터 
dataset = load_diabetes()
x = dataset.data                
y = dataset.target              

pca1 = PCA(n_components=8)
pca1 = pca1.fit_transform(x)

pca2 = PCA(n_components=10)
pca2 = pca2.fit_transform(x)

pca1_x_train, pca1_x_test, y_train, y_test = train_test_split(pca1, y, train_size = 0.8, shuffle = True, random_state = 66)
pca2_x_train, pca2_x_test, y_train, y_test = train_test_split(pca2, y, train_size = 0.8, shuffle = True, random_state = 66)

# 2. 모델 구성
model1 = Sequential()
model1.add(Dense(128, activation='relu', input_shape=(8, )))
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(1))

model2 = Sequential()
model2.add(Dense(128, activation='relu', input_shape=(10, )))
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(8, activation='relu'))
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

# loss1 :  3269.803466796875
# mse1 :  46.5020637512207
# r2 :  0.4961813837337199
# loss2 :  3241.955810546875
# mse2 :  46.01699447631836
# r2 :  0.5004721245872099
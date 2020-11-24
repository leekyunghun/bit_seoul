# 실습 pca로 축소해서 모델을 완성
# 1. 축소한 차원의 가치가 0.95 이상    -> 2
# 2. 축소한 차원의 가치가 1.0 이상     -> 4

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dropout, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical               

# 1. 데이터 
dataset = load_iris()
x = dataset.data          # (150, 4)      
y = dataset.target              

pca1 = PCA(n_components = 2)
pca1 = pca1.fit_transform(x)

pca2 = PCA(n_components = 4)
pca2 = pca2.fit_transform(x)

pca1_x_train, pca1_x_test, y_train, y_test = train_test_split(pca1, y, train_size = 0.8, shuffle = True, random_state = 66)
pca2_x_train, pca2_x_test, y_train, y_test = train_test_split(pca2, y, train_size = 0.8, shuffle = True, random_state = 66)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
model1 = Sequential()
model1.add(Dense(30, activation = 'relu', input_shape = (2, )))
model1.add(Dense(50, activation = 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(20, activation = 'relu'))
model1.add(Dense(3, activation = 'softmax'))


model2 = Sequential()
model2.add(Dense(30, activation = 'relu', input_shape = (4, )))
model2.add(Dense(50, activation = 'relu'))
model2.add(Dropout(0.2))
model2.add(Dense(20, activation = 'relu'))
model2.add(Dense(3, activation = 'softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') 

model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model1.fit(pca1_x_train, y_train, epochs = 50, batch_size = 30, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model2.fit(pca2_x_train, y_train, epochs = 50, batch_size = 30, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4. 평가, 예측
loss1, accuracy1 = model1.evaluate(pca1_x_test, y_test, batch_size = 1)
loss2, accuracy2 = model2.evaluate(pca2_x_test, y_test, batch_size = 1)

print("loss1 : ", loss1)
print("accuracy1 : ", accuracy1)
print("loss2 : ", loss2)
print("accuracy2 : ", accuracy2)

# loss1 :  0.20148062705993652
# accuracy1 :  0.9666666388511658
# loss2 :  0.13389281928539276
# accuracy2 :  1.0
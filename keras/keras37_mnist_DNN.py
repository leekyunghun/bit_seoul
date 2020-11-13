import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] * x_train.shape[2]) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2]) / 255.0

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential()
model.add(Dense(2000, activation = 'relu', input_shape = (784, )))          # Dnn 이므로 input_shape를 Dnn형식으로 맞춰줘야함
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(500, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))                                # Dnn이여도 분류모델이기 때문에 출력 수는 10개

model.summary()

#3.컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size = 32, validation_split=0.2)

#4.평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", loss)
print("accuracy : ", accuracy)

predict = model.predict(x_test)
pred = [np.argmax(predict[i]) for i in range(9400, 9420)]
print(pred)

y_test_recovery = np.argmax(y_test, axis=1).reshape(-1,1)                   # reshape(-1, 1)은 열 갯수에 맞춰서 행을 자동으로 맞춰줌
y_test_recovery = y_test_recovery.reshape(y_test_recovery.shape[1], y_test_recovery.shape[0])
print(y_test_recovery.shape)
print("y_test : ", y_test_recovery[0, 9400:9420])

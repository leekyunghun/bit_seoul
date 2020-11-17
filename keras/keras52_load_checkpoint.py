# Onehotencoding

# 1.keras
# to_categorical() 사용

# 2.sklearn
# OneHotEncoder() 사용

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255.

# 2.모델
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.models import load_model
# model = load_model("./save/model_test02_2.h5")                                        # model.fit 이후에 save를 한 model은 가중치값까지 다 가지고있다.

# model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.models import load_model
model = load_model('./model/mnist/03- 0.060129.hdf5')
model.save("./save/mnist/mnist1_model_2.h5")
model.save_weights("./save/mnist/mnist1_weights.h5")

# 4.평가, 예측
result = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", result[0])
print("accuracy : ", result[1])

# 313/313 [==============================] - 1s 2ms/step - loss: 0.0587 - accuracy: 0.9826
# loss :  0.05871664360165596
# accuracy :  0.9825999736785889
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# 2.모델
model = Sequential()
model.add(LSTM(100, activation = 'relu', input_shape = (None, 1)))          # input_shape가 항상 다르기때문에 (None, 1)로 함
model.add(Dense(100, name = "queen1"))
model.add(Dense(50, activation = 'relu', name = "queen2"))
model.add(Dense(10, name = "queen3"))
# model.add(Dense(1, name = "queen4"))

model.summary()

model.save("./save/keras28.h5")          # save파일의 확장자 = .h5
print("저장 완료!")

# model.save(".\save\keras28_2.h5")        # 파일 이름이 n으로 시작해서 \n 될수도있으니 조심
# model.save(".//save//keras28_3.h5")
# model.save(".\\save\\keras28_4.h5")

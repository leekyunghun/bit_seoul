import numpy as np

dataset = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)

    return np.array(aaa)

datasets = split_x(dataset, size)

# 1.데이터
x = datasets[: , 0:4]                   # input 슬라이싱        datasets의 1 ~ 4열에 해당하는 값들을 저장
y = datasets[: , 4]                     # output 슬라이싱       datasets의 5열에 해당하는 값들을 저장

x_input = np.array([7, 8, 9, 10])       # test input

x = x.reshape(6, 4, 1)                  # LSTM input_shape 맞추기   
x_input = x_input.reshape(1, 4, 1)      # LSTM input_shape 맞추기

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input = Input(shape=(4,1))
lstm = LSTM(30, activation = 'relu')(input) 
dense1 = Dense(50)(lstm)
dense2 = Dense(30, activation = 'relu')(dense1)
dense3 = Dense(10)(dense2)
output = Dense(1)(dense3)

model = Model(inputs = input, outputs = output)
model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 40, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 300, batch_size = 1, verbose = 1)

# 4.평가, 예측
predict = model.predict(x_input)
print("predict : ", predict)
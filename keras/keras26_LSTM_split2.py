import numpy as np

dataset = np.array(range(1, 101))  
size = 5

# 실습 : 모델 구성
# train, test 분리
# early_stopping

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)

    return np.array(aaa)

datasets = split_x(dataset, size)
print(dataset.shape)

# 1.데이터
x = datasets[: , 0:4]                   # input 슬라이싱        datasets의 1 ~ 4열에 해당하는 값들을 저장
y = datasets[: , 4]                     # output 슬라이싱       datasets의 5열에 해당하는 값들을 저장


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)      # 주어진 데이터전체에서 train, test set을 자동으로 만들어줌

x_train = x_train.reshape(76, 4, 1)
x_test = x_test.reshape(20, 4, 1)
x_pred = np.array([97, 98, 99, 100])
x_pred = x_pred.reshape(1, 4, 1)

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
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 300, batch_size = 1, validation_split = 0.2, verbose = 1)

# 4.평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mse : ", mse)

y_predict = model.predict(x_pred)
print("y_predict : ", y_predict)
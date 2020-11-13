import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,LSTM

dataset = np.array(range(1, 101))  
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle=True)      # 주어진 데이터전체에서 train, test set을 자동으로 만들어줌

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2.모델
model = load_model("./save/keras28.h5")
# ValueError: All layers added to a Sequential model should have unique names.
# Name "dense" is already the name of a layer in this model. Update the `name` argument to pass a unique name.      # model을 load해오고 그냥 Dense를 추가했을때 나오는 에러

model.add(Dense(5, activation = 'relu', name = 'king1'))
model.add(Dense(1, name = 'king2'))
model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 300, batch_size = 10, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4.평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mse : ", mse)

predict = model.predict(x_test)
print("predict : ", predict)

print(y_test)
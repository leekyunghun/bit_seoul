import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Concatenate

# 데이터 split
def split_x_2(seq, size):                           
    bbb = []
    for i in range(len(seq) - size + 1):
        for j in range(i, i+1):
            bbb.append(seq[j:j+size, :])
    bbb = np.array(bbb)
    return bbb

samsung_x_data = np.load('./data/samsung_x_data_npy.npy', allow_pickle=True).astype('float32')
samsung_y_data = np.load('./data/samsung_y_data_npy.npy', allow_pickle=True).astype('float32')
samsung_predict_data = np.load('./data/samsung_predict_data_npy.npy', allow_pickle=True).astype('float32')

bit_computer_x_data = np.load('./data/bit_computer_x_data_npy.npy', allow_pickle=True).astype('float32')
bit_computer_predict_data = np.load('./data/bit_computer_predict_data_npy.npy', allow_pickle=True).astype('float32')

# 1.데이터
samsung_y_data = samsung_y_data.reshape(659, 1)

samsung_x_data_train, samsung_x_data_test, samsung_y_data_train, samsung_y_data_test = train_test_split(samsung_x_data, samsung_y_data, train_size = 0.7)
bit_computer_x_data_train, bit_computer_x_data_test = train_test_split(bit_computer_x_data, train_size = 0.7)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler       # 데이터 전처리 기능 standardScaler 사용
# scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler1 = StandardScaler()
scaler2 = StandardScaler()

scaler1.fit(samsung_x_data)                                                   
samsung_x_data = scaler1.transform(samsung_x_data)                                       

scaler2.fit(bit_computer_x_data)  
bit_computer_x_data = scaler2.transform(bit_computer_x_data)                                       

samsung_x_data_train = split_x_2(samsung_x_data_train,5)                            # 데이터 split함수를 사용
samsung_x_data_test = split_x_2(samsung_x_data_test, 5)
samsung_y_data_train = split_x_2(samsung_y_data_train, 5)
samsung_y_data_test = split_x_2(samsung_y_data_test, 5)

bit_computer_x_data_train = split_x_2(bit_computer_x_data_train, 5)
bit_computer_x_data_test = split_x_2(bit_computer_x_data_test, 5)

# (457, 5, 6) (194, 5, 6) (457, 5, 1) (194, 5, 1)       # train shape
# (457, 5, 5) (457, 5, 1)                               # test shape
# (1, 5, 6) (1, 5, 5)                                   # x_predict shape

samsung_predict_data = samsung_predict_data.reshape(1, 1, 6)
bit_computer_predict_data = bit_computer_predict_data.reshape(1, 1, 5)

# 2.모델 구성

# # 모델1
# input1 = Input(shape=(5, 6)) 
# dense1_1 = Dense(100, activation='relu')(input1)
# dense1_2 = Dense(30, activation='relu')(dense1_1) 
# dense1_3 = Dense(7, activation='relu')(dense1_2)
# output1 = Dense(1)(dense1_3)

# # 모델2
# input2 = Input(shape=(5, 5)) 
# dense2_1 = Dense(100, activation='relu')(input2) 
# dense2_2= Dense(30, activation='relu')(dense2_1) 
# dense2_3= Dense(7, activation='relu')(dense2_2)
# output2 = Dense(1)(dense2_3)

# #모델 병합
# merge1 = Concatenate()([output1, output2]) 

# middle1 = Dense(30)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(11)(middle2)

# #output 모델 구성 
# output = Dense(30)(middle3)
# output = Dense(13)(output)
# output = Dense(7)(output)
# output = Dense(1)(output)

# model = Model(inputs = [input1, input2], outputs = output)
# model.summary()

model = load_model('./model/samsung_bitcomputer-checkpoint.hdf5')         # CheckPoint 불러오기
# model.save("./save/samsung_bitcomputer_model.h5")                         # 불러온 CheckPoint의 모델 저장
# model.save_weights("./save/samsung_bitcomputer_weight.h5")                # 불러온 CheckPoint의 가중치 저장
# model.load_weights("./save/samsung_bitcomputer_weight.h5")                # 저장한 가중치 불러오기             

# # 3.컴파일, 훈련
# modelpath = "./model/homework-{epoch:02d}-{val_loss: 4f}-01.hdf5"

# from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1000, mode = 'min')
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
# model.fit([samsung_x_data_train, bit_computer_x_data_train], samsung_y_data_train, epochs = 10000, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping, cp])

# # 4.평가, 예측
# loss, mse = model.evaluate([samsung_x_data_test, bit_computer_x_data_test], samsung_y_data_test)
# print("loss : ", loss)
# print("mse : ", mse)

predict = model.predict([samsung_predict_data, bit_computer_predict_data])
print("predict : ", predict)
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

# numpy 데이터 불러오기
samsung_x_data = np.load('./data/homework3/samsung_x_data_npy_2.npy', allow_pickle=True).astype('float32')
samsung_y_data = np.load('./data/homework3/samsung_y_data_npy_2.npy', allow_pickle=True).astype('float32')
samsung_predict_data = np.load('./data/homework3/samsung_predict_data_npy_2.npy', allow_pickle=True).astype('float32')

bit_computer_x_data = np.load('./data/homework3/bit_computer_x_data_npy_2.npy', allow_pickle=True).astype('float32')
bit_computer_predict_data = np.load('./data/homework3/bit_computer_predict_data_npy_2.npy', allow_pickle=True).astype('float32')

gold_x_data = np.load('./data/homework3/gold_x_data_npy_2.npy', allow_pickle=True).astype('float32')
gold_predict_data = np.load('./data/homework3/gold_predict_data_npy_2.npy', allow_pickle=True).astype('float32')

kosdaq_x_data = np.load('./data/homework3/kosdaq_x_data_npy_2.npy', allow_pickle=True).astype('float32')
kosdaq_predict_data = np.load('./data/homework3/kosdaq_predict_data_npy_2.npy', allow_pickle=True).astype('float32')

#1.데이터
samsung_y_data = samsung_y_data.reshape(659, 1)

# scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()

scaler1.fit(samsung_x_data)                                                   
samsung_x_data = scaler1.transform(samsung_x_data)                                       

scaler2.fit(bit_computer_x_data)  
bit_computer_x_data = scaler2.transform(bit_computer_x_data)  

scaler3.fit(gold_x_data)                                                   
gold_x_data = scaler3.transform(gold_x_data)    

scaler4.fit(kosdaq_x_data)                                                   
kosdaq_x_data = scaler4.transform(kosdaq_x_data)  

# train_test_split
samsung_x_data_train, samsung_x_data_test, samsung_y_data_train, samsung_y_data_test = train_test_split(samsung_x_data, samsung_y_data, train_size = 0.7)

bit_computer_x_data_train, bit_computer_x_data_test = train_test_split(bit_computer_x_data, train_size = 0.7)

gold_x_data_train, gold_x_data_test = train_test_split(gold_x_data, train_size = 0.7)

kosdaq_x_data_train, kosdaq_x_data_test = train_test_split(kosdaq_x_data, train_size = 0.7)

# 데이터 split함수를 사용
samsung_x_data_train = split_x_2(samsung_x_data_train,5)                            
samsung_x_data_test = split_x_2(samsung_x_data_test, 5)
samsung_y_data_train = split_x_2(samsung_y_data_train, 5)
samsung_y_data_test = split_x_2(samsung_y_data_test, 5)

bit_computer_x_data_train = split_x_2(bit_computer_x_data_train, 5)
bit_computer_x_data_test = split_x_2(bit_computer_x_data_test, 5)

gold_x_data_train = split_x_2(gold_x_data_train, 5)
gold_x_data_test = split_x_2(gold_x_data_test, 5)

kosdaq_x_data_train = split_x_2(kosdaq_x_data_train, 5)
kosdaq_x_data_test = split_x_2(kosdaq_x_data_test, 5)

print(samsung_x_data_train.shape, bit_computer_x_data_train.shape, gold_x_data_train.shape, kosdaq_x_data_train.shape)
print(samsung_x_data_test.shape, bit_computer_x_data_test.shape, gold_x_data_test.shape, kosdaq_x_data_test.shape)
print(samsung_y_data_train.shape, samsung_y_data_test.shape)
print(samsung_predict_data.shape, bit_computer_predict_data.shape, gold_predict_data.shape, kosdaq_predict_data.shape)
# (457, 5, 3) (457, 5, 6) (457, 5, 4) (457, 5, 5)   train shape
# (194, 5, 3) (194, 5, 6) (194, 5, 4) (194, 5, 5)   test shape
# (457, 5, 1) (194, 5, 1)                           y_data shape

samsung_predict_data = samsung_predict_data.reshape(1, 3)
bit_computer_predict_data = bit_computer_predict_data.reshape(1, 6)
gold_predict_data = gold_predict_data.reshape(1, 4)
kosdaq_predict_data = kosdaq_predict_data.reshape(1, 5)

# 2.모델 구성

# 모델1
input1 = Input(shape=(5, 3)) 
dense1_1 = Dense(100, activation='relu')(input1)
dense1_2 = Dense(50, activation='relu')(dense1_1) 
dense1_3 = Dense(10, activation='relu')(dense1_2)
output1 = Dense(1)(dense1_3)

# 모델2
input2 = Input(shape=(5, 6)) 
dense2_1 = Dense(100, activation='relu')(input2) 
dense2_2 = Dense(50, activation='relu')(dense2_1) 
dense2_3 = Dense(10, activation='relu')(dense2_2)
output2 = Dense(1)(dense2_3)

# 모델3
input3 = Input(shape=(5, 4)) 
dense3_1 = Dense(100, activation='relu')(input3) 
dense3_2 = Dense(50, activation='relu')(dense3_1) 
dense3_3 = Dense(10, activation='relu')(dense3_2) 
output3 = Dense(1)(dense3_3)

# 모델4
input4 = Input(shape=(5, 5))
dense4_1 = Dense(100, activation='relu')(input4) 
dense4_2 = Dense(50, activation='relu')(dense4_1)
dense4_3 = Dense(10, activation='relu')(dense4_2)
output4 = Dense(1)(dense4_3)

#모델 병합
merge1 = Concatenate()([output1, output2, output3, output4]) 
middle1 = Dense(100, activation='relu')(merge1)
middle1 = Dropout(0.2)(middle1)
middle2 = Dense(30, activation='relu')(middle1)

#output 모델 구성 (분기-나눔)
output = Dense(100, activation='relu')(middle2)
output = Dropout(0.2)(output)
output = Dense(30)(output)
output = Dense(4, activation='relu')(output)
output = Dense(1)(output)

model = Model(inputs = [input1, input2, input3, input4], outputs = output)

# model = load_model('./model/homework3/1592- 324243849216.000000-05.hdf5')         # CheckPoint 불러오기
# model.save("./save/homework3/homework_model.h5")                         # 불러온 CheckPoint의 모델 저장
# model.save_weights("./save/homework3/homework_weight.h5")                # 불러온 CheckPoint의 가중치 저장
# model.load_weights("./save/samsung_bitcomputer_weight.h5")                # 저장한 가중치 불러오기     

# # 3.컴파일, 훈련
# modelpath = "./model/homework3/{epoch:02d}-{val_loss: 4f}-05.hdf5"

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3000, mode = 'min')
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([samsung_x_data_train, bit_computer_x_data_train, gold_x_data_train, kosdaq_x_data_train], samsung_y_data_train, epochs = 10000, batch_size = 50, validation_split = 0.3, verbose = 1, callbacks = [early_stopping, cp])

# 4.평가, 예측
loss, mse = model.evaluate([samsung_x_data_test, bit_computer_x_data_test, gold_x_data_test, kosdaq_x_data_test], samsung_y_data_test)
print("loss : ", loss)
print("mse : ", mse)

predict = model.predict([samsung_predict_data, bit_computer_predict_data, gold_predict_data, kosdaq_predict_data])
print("predict : ", predict)
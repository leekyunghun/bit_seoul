import numpy as np
import pandas as pd

# 데이터 split
def split_x_2(seq, size):                           
    bbb = []
    for i in range(len(seq) - size + 1):
        for j in range(i, i+1):
            bbb.append(seq[j:j+size, :])
    bbb = np.array(bbb)
    return bbb

# 데이터 불러오기
samsung = pd.read_csv("./data/csv/삼성전자 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')          # (660, 17)
bit_computer = pd.read_csv("./data/csv/비트컴퓨터 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')   # (660, 17)
bit_computer_slice = bit_computer[0:660]

# 등락률을 보고 전일비 데이터 빈공간 채우기
def changeUp0rDown(array):
    for i in range(len(array)):
        if array.iloc[i, 5] == '▲':
            array.iloc[i, 5] = '1,'
        elif array.iloc[i, 5] == '▼':
            array.iloc[i, 5] = '0,'
        else:
            if array.iloc[i, 7] > 0:
                array.iloc[i, 5] = '1,'
            else:
                array.iloc[i, 5] = '0,'
    return array

# 전일비 빈공간을 채워줌
samsung = changeUp0rDown(samsung)
bit_computer = changeUp0rDown(bit_computer_slice)

samsung = samsung.sort_values(['일자'], ascending=['True'])
bit_computer = bit_computer.sort_values(['일자'], ascending=['True'])

# 삼성주식 627일 ~ 629일의 거래량과 금액이 없으므로 0을 주었음 
samsung.iloc[30:33, 8:10] = '0'

# 값들을 문자열에서 정수로 변경
for i in range(len(samsung.index)):
    samsung.iloc[i,1] = int(samsung.iloc[i,1].replace(',',''))
    samsung.iloc[i,2] = int(samsung.iloc[i,2].replace(',',''))
    samsung.iloc[i,3] = int(samsung.iloc[i,3].replace(',',''))
    samsung.iloc[i,4] = int(samsung.iloc[i,4].replace(',',''))
    samsung.iloc[i,5] = int(samsung.iloc[i,5].replace(',',''))
    samsung.iloc[i,8] = int(samsung.iloc[i,8].replace(',',''))
    samsung.iloc[i,9] = int(samsung.iloc[i,9].replace(',',''))
 
for i in range(len(bit_computer.index)):
    bit_computer.iloc[i,1] = int(bit_computer.iloc[i,1].replace(',',''))
    bit_computer.iloc[i,2] = int(bit_computer.iloc[i,2].replace(',',''))
    bit_computer.iloc[i,3] = int(bit_computer.iloc[i,3].replace(',',''))
    bit_computer.iloc[i,4] = int(bit_computer.iloc[i,4].replace(',',''))
    bit_computer.iloc[i,5] = int(bit_computer.iloc[i,5].replace(',',''))
    bit_computer.iloc[i,8] = int(bit_computer.iloc[i,8].replace(',',''))

samsung_choice_data = samsung[['시가', '고가', '저가','종가', '전일비', '거래량','금액(백만)']]
samsung_choice_data = samsung_choice_data.iloc[:-1, :]

samsung_x_data = samsung_choice_data[['시가', '고가', '저가', '전일비', '거래량','금액(백만)']]
samsung_y_data = samsung_choice_data['종가']
samsung_predict_data = samsung_x_data.iloc[-1, :]

bit_computer_choice_data = bit_computer[['시가', '고가', '저가', '전일비', '거래량']]
bit_computer_x_data = bit_computer_choice_data.iloc[:-1, :]
bit_computer_predict_data = bit_computer_x_data.iloc[-1, :]

samsung_x_data.to_csv("./data/csv/samsung_x_data.csv", mode='w')
samsung_y_data.to_csv("./data/csv/samsung_y_data.csv", mode='w')
samsung_predict_data.to_csv("./data/csv/samsung_predict_data.csv", mode='w')

bit_computer_x_data.to_csv("./data/csv/bit_computer_x_data.csv", mode='w')
bit_computer_predict_data.to_csv("./data/csv/bit_computer_predict_data.csv", mode='w')


samsung_x_data_npy = samsung_x_data.values
samsung_y_data_npy = samsung_y_data.values
samsung_predict_data_npy = samsung_predict_data.values

bit_computer_x_data_npy = bit_computer_x_data.values
bit_computer_predict_data_npy = bit_computer_predict_data.values

np.save('./data/samsung_x_data_npy.npy', arr = samsung_x_data_npy)
np.save('./data/samsung_y_data_npy.npy', arr = samsung_y_data_npy)
np.save('./data/samsung_predict_data_npy.npy', arr = samsung_predict_data_npy)

np.save('./data/bit_computer_x_data_npy.npy', arr = bit_computer_x_data_npy)
np.save('./data/bit_computer_predict_data_npy.npy', arr = bit_computer_predict_data_npy)

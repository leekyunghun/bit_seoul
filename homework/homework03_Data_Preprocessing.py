import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/삼성전자 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')         
bit_computer = pd.read_csv("./data/csv/비트컴퓨터 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  
gold = pd.read_csv("./data/csv/금현물.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  
kosdaq = pd.read_csv("./data/csv/코스닥.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  

bit_computer_slice = bit_computer[0:660].copy()
gold_slice = gold[0:660].copy()
kosdaq_slice = kosdaq[0:660].copy()

# 등락률을 보고 전일비 데이터 빈공간 채우기
def changeUp0rDown(array):
    for i in range(len(array)):
        if array.iloc[i, 5] == '▲':
            array.iloc[i, 5] = '1,'
        elif array.iloc[i, 5] == '▼':
            array.iloc[i, 5] = '0,'
        else:
            if int(array.iloc[i, 7]) > 0:
                array.iloc[i, 5] = '1,'
            else:
                array.iloc[i, 5] = '0,'
    return array

# 등락률을 보고 전일비의 형태를 1과 0으로 변경 (등락률이 양수값이면 1, 음수값이면 0)
for i in range(len(gold_slice.index)):
    gold_slice.iloc[i,7] = gold_slice.iloc[i, 7].replace("%", "")

bit_computer_slice = changeUp0rDown(bit_computer_slice)
gold_slice = changeUp0rDown(gold_slice)
kosdaq_slice = changeUp0rDown(kosdaq_slice)

# 오름차순으로 정렬
samsung = samsung.sort_values(['일자'], ascending=['True'])
bit_computer_slice = bit_computer_slice.sort_values(['일자'], ascending=['True'])
gold_slice = gold_slice.sort_values(['일자'], ascending=['True'])
kosdaq_slice = kosdaq_slice.sort_values(['일자'], ascending=['True'])

# 삼성주식의 627일 부터 629일까지의 거래량이 없어서 0으로 설정
samsung.iloc[30:33, 8] = "0"

def changeValueType(input_array, colums):
    for i in range(len(input_array)):
        for j in colums: 
            input_array.iloc[i,j] = float(input_array.iloc[i,j].replace(',',''))
    return input_array

# 내가 정한 colum값의 type이 str인 부분을 float 형태로 변환
changeValueType(samsung, [1,4,8])         
changeValueType(bit_computer_slice, [2,3,4,5,8])
changeValueType(gold_slice, [4,5,7,8])
changeValueType(kosdaq_slice,[5,8,9])

# input 데이터, output데이터, predict data 선언
samsung_choice_data = samsung[['시가','종가', '등락률', '거래량']]
samsung_choice_data = samsung_choice_data.iloc[:-1, :]

samsung_x_data = samsung_choice_data[['종가', '등락률', '거래량']]      # 2020/11/19일 까지의 데이터
samsung_y_data = samsung_choice_data['시가']                           # 2020/11/19일 까지의 데이터
samsung_predict_data = samsung_x_data.iloc[-1, :]

bit_computer_choice_data = bit_computer_slice[['종가', '고가', '저가', '전일비', '등락률', '거래량']]
bit_computer_x_data = bit_computer_choice_data.iloc[:-1, :]
bit_computer_predict_data = bit_computer_x_data.iloc[-1, :]

gold_choice_data = gold_slice[['종가', '전일비', '등락률', '거래량']]    # 2020/11/20일 까지의 데이터
gold_x_data = gold_choice_data.iloc[:-1, :]
gold_predict_data = gold_x_data.iloc[-1, :]

kosdaq_choice_data = kosdaq_slice[['현재가', '전일대비', '등락률', '거래량', '거래대금']]
kosdaq_x_data = kosdaq_choice_data.iloc[:-1, :]
kosdaq_predict_data = kosdaq_x_data.iloc[-1, :]

# 나눠놓은 데이터들을 csv파일로 저장
samsung_x_data.to_csv("./data/homework3/samsung_x_data_2.csv", mode='w')
samsung_y_data.to_csv("./data/homework3/samsung_y_data_2.csv", mode='w')
samsung_predict_data.to_csv("./data/homework3/samsung_predict_data_2.csv", mode='w')

bit_computer_x_data.to_csv("./data/homework3/bit_computer_x_data_2.csv", mode='w')
bit_computer_predict_data.to_csv("./data/homework3/bit_computer_predict_data_2.csv", mode='w')

gold_x_data.to_csv("./data/homework3/gold_x_data_2.csv", mode='w')
gold_predict_data.to_csv("./data/homework3/gold_predict_data_2.csv", mode='w')

kosdaq_x_data.to_csv("./data/homework3/kosdaq_x_data_2.csv", mode='w')
kosdaq_predict_data.to_csv("./data/homework3/kosdaq_predict_data_2.csv", mode='w')

# pandas 내용을 numpy로 변형
samsung_x_data_npy = samsung_x_data.values
samsung_y_data_npy = samsung_y_data.values
samsung_predict_data_npy = samsung_predict_data.values

bit_computer_x_data_npy = bit_computer_x_data.values
bit_computer_predict_data_npy = bit_computer_predict_data.values

gold_x_data_npy = gold_x_data.values
gold_predict_data_npy = gold_predict_data.values

kosdaq_x_data_npy = kosdaq_x_data.values
kosdaq_predict_data_npy = kosdaq_predict_data.values

# numpy형태인 데이터들을 저장
np.save('./data/homework3/samsung_x_data_npy_2.npy', arr = samsung_x_data_npy)
np.save('./data/homework3/samsung_y_data_npy_2.npy', arr = samsung_y_data_npy)
np.save('./data/homework3/samsung_predict_data_npy_2.npy', arr = samsung_predict_data_npy)

np.save('./data/homework3/bit_computer_x_data_npy_2.npy', arr = bit_computer_x_data_npy)
np.save('./data/homework3/bit_computer_predict_data_npy_2.npy', arr = bit_computer_predict_data_npy)

np.save('./data/homework3/gold_x_data_npy_2.npy', arr = gold_x_data_npy)
np.save('./data/homework3/gold_predict_data_npy_2.npy', arr = gold_predict_data_npy)

np.save('./data/homework3/kosdaq_x_data_npy_2.npy', arr = kosdaq_x_data_npy)
np.save('./data/homework3/kosdaq_predict_data_npy_2.npy', arr = kosdaq_predict_data_npy)

print("데이터 저장 완료!!")
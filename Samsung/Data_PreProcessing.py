import numpy as np
import pandas as pd

samsung = pd.read_csv("./data/csv/삼성전자 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')         
bit_computer = pd.read_csv("./data/csv/비트컴퓨터 1120.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  
gold = pd.read_csv("./data/csv/금현물.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  
kosdaq = pd.read_csv("./data/csv/코스닥.csv", header = 0, index_col = None, sep = ',', encoding = 'CP949')  

bit_computer = bit_computer.iloc[:660, :]
gold = gold.iloc[:660, :]
kosdaq = kosdaq.iloc[:660, :]

# 오름차순으로 정렬
samsung = samsung.sort_values(['일자'], ascending=['True'])
bit_computer = bit_computer.sort_values(['일자'], ascending=['True'])
gold = gold.sort_values(['일자'], ascending=['True'])
kosdaq = kosdaq.sort_values(['일자'], ascending=['True'])

samsung = samsung[['시가','고가', '저가', '종가']]
samsung_x = samsung[['고가', '저가', '종가']]
samsung_x = samsung_x.iloc[:-1, :]
samsung_y = samsung['시가']
samsung_y = samsung_y.iloc[1:]   # (659, )   # 2020/11/20 data를 이용하여 23일의 시가를 맞춰야하므로 1일씩 미뤄서 y값으로 해놓음
samsung_predict = samsung_x.iloc[-1, :] # (3, )

bit_computer = bit_computer[['고가', '저가', '거래량', '금액(백만)', '개인', '기관']]   # (660, 6)
bit_computer_x = bit_computer.iloc[:-1, :]
bit_computer_predict = bit_computer.iloc[-1, :] # (6, )

gold = gold[['고가', '저가', '종가', '거래량', '거래대금(백만)']]  # (660, 5)
gold_x = gold.iloc[:-1, :]
gold_predict = gold.iloc[-1, :]  # (4, )

kosdaq = kosdaq[['고가', '저가', '현재가', '상승']]  # (660, 4)
kosdaq_x = kosdaq.iloc[:-1, :]
kosdaq_predict = kosdaq.iloc[-1, :] # (5, )

samsung_x = samsung_x.values
samsung_y = samsung_y.values
samsung_predict = samsung_predict.values

bit_computer_x = bit_computer_x.values
bit_computer_predict = bit_computer_predict.values

gold_x = gold_x.values
gold_predict = gold_predict.values

kosdaq_x = kosdaq_x.values
kosdaq_predict = kosdaq_predict.values

for i in range(samsung_x.shape[0]):
    for j in range(samsung_x.shape[1]): 
        samsung_x[i,j] = float(samsung_x[i,j].replace(',',''))

for i in range(samsung_y.shape[0]):
    samsung_y[i] = float(samsung_y[i].replace(',',''))

for i in range(bit_computer_x.shape[0]):
    for j in range(bit_computer_x.shape[1]): 
        bit_computer_x[i,j] = float(bit_computer_x[i,j].replace(',',''))

for i in range(bit_computer_predict.shape[0]):
    bit_computer_predict[i] = float(bit_computer_predict[i].replace(',',''))

for i in range(gold_x.shape[0]):
    for j in range(gold_x.shape[1]): 
        gold_x[i,j] = float(gold_x[i,j].replace(',',''))

for i in range(gold_predict.shape[0]):
    gold_predict[i] = float(gold_predict[i].replace(',',''))

for i in range(kosdaq_x.shape[0]):
  kosdaq_x[i,3] = float(kosdaq_x[i,3].replace(',',''))

kosdaq_predict[3] = float(kosdaq_predict[3].replace(',',''))

# np.save("./Samsung/samsung_x.npy", arr = samsung_x)
# np.save("./Samsung/samsung_y.npy", arr = samsung_y)
# np.save("./Samsung/samsung_predict.npy", arr = samsung_predict)

# np.save("./Samsung/bit_computer_x.npy", arr = bit_computer_x)
# np.save("./Samsung/bit_computer_predict.npy", arr = bit_computer_predict)

# np.save("./Samsung/gold_x.npy", arr = gold_x)
# np.save("./Samsung/gold_predict.npy", arr = gold_predict)

# np.save("./Samsung/kosdaq_x.npy", arr = kosdaq_x)
# np.save("./Samsung/kosdaq_predict.npy", arr = kosdaq_predict)


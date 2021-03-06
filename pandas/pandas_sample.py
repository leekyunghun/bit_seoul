import pandas as pd
import numpy as np

from numpy.random import randn
np.random.seed(100)

data = randn(5, 4)
# print(data)

df = pd.DataFrame(data, index = 'A B C D E'.split(), columns = '가 나 다 라'.split())
# print(df)

data2 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]          # (5, 4)
df2 = pd.DataFrame(data2, index = ['A', 'B', 'C', 'D', 'E'], columns = ['가', '나', '다', '라'])
# print(df2)

df3 = pd.DataFrame(np.array([[1,2,3], [4,5,6]]))
# print(df3)

#     가   나   다   라
# A   1   2   3   4
# B   5   6   7   8
# C   9  10  11  12
# D  13  14  15  16
# E  17  18  19  20

# colum(열)
print("df2['나'] : \n", df2['나'])
# df2['나'] : 
#   A     2
#   B     6
#   C    10
#   D    14
#   E    18

print("df2['나', '라]: \n", df2[['나', '라']])
# df2['나', '라]: 
#      나   라
#   A   2   4
#   B   6   8
#   C  10  12
#   D  14  16
#   E  18  20

# # print("df2[0] : ", df2[0])                               # 에러
# # print("df2.loc['나'] : ", df2.loc['나'])                 # loc에 colum명이 들어가면 안된다. -> row명만 가능

print("df2.iloc[:, 2] : \n", df2.iloc[:, 2])               # iloc의 i = index ,   df2.iloc[:, 2]은 모든 행의 2번째 인덱스를 말한다.
# df2.iloc[:, 2] : 
#   A     3
#   B     7
#   C    11
#   D    15
#   E    19

# print("df2[:, 2] : \n", df2[:, 2])                       # numpy에서나 가능한 방식 pandas에서는 불가능

# # row(행)
print("df2.loc['A'] : ", df2.loc['A'])                     # loc는 row명으로 사용가능 
# df2.loc['A'] : 
# 가    1
# 나    2
# 다    3
# 라    4

print("df2.loc['A', 'C'] : ", df2.loc[['A', 'C']])
# df2.loc['A', 'C'] :     가   나   다   라
#                      A   1   2    3    4
#                      C   9   10   11   12

print("df2.iloc[0] : ", df2.iloc[0])                     # loc는 row명으로 사용가능 
# df2.iloc[0] :  
# 가    1
# 나    2
# 다    3
# 라    4

print("df2.iloc[[0, 2]] : ", df2.iloc[[0, 2]])
# df2.iloc[[0, 2]] :     
#   가    나    다    라
# A  1    2     3     4
# C  9   10    11    12

# 행렬
print("df2.loc[['A', 'B']], [['나', '다']] : \n", df2.loc[['A', 'B'], ['나', '다']])
# df2.loc[['A', 'B']], [['나', '다']] : 
#       나   다
#   A    2   3 
#   B    6   7

# 한개의 값만 확인
print("df2.loc['E', '다']", df2.loc['E', '다'])
# df2.loc['E', '다'] 19

print("df2.iloc[4,2] : ", df2.iloc[4,2])
# df2.iloc[4,2] :  19

print("df2.iloc[4][2] : ", df2.iloc[4][2])
# df2.iloc[4][2] :  19
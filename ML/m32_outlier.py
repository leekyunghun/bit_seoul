
############################### 데이터 안에 있는 이상치 확인 ################################

import numpy as np

# def outliers(data_out):
#     quartile_1, quartile_3 = np.percentile(data_out, [25, 75])
#     print("1사분위 : ", quartile_1)     # 3.25
#     print("3사분위 : ", quartile_3)     # 97.5
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)

#     return np.where((data_out > upper_bound) | (data_out < lower_bound))

# a = np.array([1, 2, 3, 4, 10000, 6, 7, 5000, 90, 100])

# b = outliers(a)
# print("이상치의 위치 : ", b)

# # 과제 2차원 배열 이상이 input일때를 생각해서 2차원 배열일때 outliers가 기능 할 수 있도록 수정

a = np.array([[1,2,3,4,10000,6,7,5000,90,100],
              [10000,20000,3,40000,50000,60000,70000,8,90000,100000]])
a = a.transpose()
print(a)
def outliers(data_out):
    result = []
    for i in range(data_out.shape[1]):
        quartile_1, quartile_3 = np.percentile(data_out[:, i], [25, 75])
        
        print("==========", i+1, "열==========")
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3, "\n")

        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print("upper_bound : ", upper_bound , "lower_bound : ", lower_bound)

        a = np.where((data_out[:, i] > upper_bound) | (data_out[:, i] < lower_bound))

        a = np.array(a)
        if a.shape[1] == 0:
             a = np.append(a, 0)
             a = a.reshape(1, a.shape[0])
        result.append(a)

    return result
b = outliers(a)

print("열 마다 이상치의 위치 : ", b)

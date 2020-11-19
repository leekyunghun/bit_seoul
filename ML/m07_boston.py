import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1.데이터
x, y = load_boston(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# 2.모델
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier()
# model = KNeighborsRegressor()
# model = RandomForestClassifier()
# model = RandomForestRegressor()

# 3.컴파일, 훈련
model.fit(x_train, y_train)

# 4 평가, 예측

score = model.score(x_test, y_test)     # model.evaluate 와 같은 기능
print("model.score : ", score)

# accuracy_score와 비교할것 -> 분류모델
# r2_score와 비교할것 -> 회귀모델

y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)           
print("R2: ", r2)

print(y_test, "의 예측 결과 : \n", y_predict[:10])

# 1. KNeighborsRegressor()
# model.score :  0.8404010032786686
# R2:  0.8404010032786686

# 2. RandomForestRegressor()
# model.score :  0.9198829921882885
# R2:  0.9198829921882885
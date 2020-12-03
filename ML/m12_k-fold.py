import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

iris = pd.read_csv('./data/csv/iris_ys.csv', header = 0, index_col=0)

x = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

print(iris)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

# 2.모델
kfold = KFold(n_splits = 5, shuffle=True)         # n_splits = validation을 몇개로 나눠서 할지

model = SVC(verbose=1)

scores = cross_val_score(model, x_train, y_train, cv = kfold)       # cross_validation에서 단계마다 나오는 scroe

print('scores : ', scores)

# 3.컴파일, 훈련
model.fit(x_train, y_train)

# 4 평가, 예측
score = model.score(x_test, y_test)     # model.evaluate 와 같은 기능
print("model.score : ", score)

# # accuracy_score와 비교할것 -> 분류모델
# # r2_score와 비교할것 -> 회귀모델

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

# from sklearn.metrics import r2_score
# r2 = r2_score(y_test, y_predict)           
# print("R2: ", r2)


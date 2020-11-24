# 와인

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'n_estimators' : [200, 300]},
    {'max_depth' : [10, 12, 16, 23]},
    {'min_samples_leaf' : [7, 11, 13, 17]},
    {'min_samples_split' : [4, 6, 10, 20]},
    {'n_jobs' : [-1]}                                       # n_jobs = 병렬로 실행할 작업 수 (-1이면 전부 다 사용)
]


# 1.데이터
wine = pd.read_csv("./data/csv/winequality-white.csv", header = 0, index_col = None, sep = ';')
wine = wine.values

# 1. 데이터
x = wine[:, :-1]
y = wine[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

print(x_train.shape, y_train.shape)

# 2. 모델
kfold = KFold(n_splits = 5, shuffle=True)
# model = SVC()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)                    # n_jobs = 병렬로 실행할 작업 수 (-1이면 전부 다 사용)

# 3.훈련
model.fit(x_train, y_train)

# 4.평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict = model.predict(x_test)
print("최종정답률 : ", accuracy_score(y_test, y_predict))
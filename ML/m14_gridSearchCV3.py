# 당뇨병 
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.utils.testing import all_estimators
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

parameters = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 11]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1]}                                       # n_jobs = 병렬로 실행할 작업 수 (-1이면 전부 다 사용)
]

# 1.데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

print(x_train.shape, y_train.shape)

# 2. 모델
kfold = KFold(n_splits = 5, shuffle=True)
# model = SVC()
model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold)

# 3.훈련
model.fit(x_train, y_train)

# 4.평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)
y_predict = model.predict(x_test)
print("최종정답률 : ", r2_score(y_test, y_predict))
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline                                        # pipeline import할 내용
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

parameters = [
    {'RandomForestClassifier__n_estimators' : [100, 200]},
    {'RandomForestClassifier__max_depth' : [10, 12, 16, 23]},
    {'RandomForestClassifier__min_samples_leaf' : [7, 11, 13, 17]},
    {'RandomForestClassifier__min_samples_split' : [4, 6, 10, 20]},
    {'RandomForestClassifier__n_jobs' : [-1]}                                       # n_jobs = 병렬로 실행할 작업 수 (-1이면 전부 다 사용)
]


# 1.데이터
wine = pd.read_csv("./data/csv/winequality-white.csv", header = 0, index_col = None, sep = ';')
wine = wine.values

# 1. 데이터
x = wine[:, :-1]
y = wine[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

# 2.모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('RandomForestClassifier', RandomForestClassifier())])    
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())                          
model = RandomizedSearchCV(pipe, parameters, cv=4)                                 

# 3.훈련
model.fit(x_train, y_train)

# 4.평가
print("acc : ", model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 parameter : ", model.best_params_)
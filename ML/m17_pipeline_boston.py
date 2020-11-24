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
    {'RandomForestRegressor__n_estimators' : [100, 200]},
    {'RandomForestRegressor__max_depth' : [6, 8, 10, 11]},
    {'RandomForestRegressor__min_samples_leaf' : [3, 5, 7, 10]},
    {'RandomForestRegressor__min_samples_split' : [2, 3, 5, 10]},
    {'RandomForestRegressor__n_jobs' : [-1]}                                       # n_jobs = 병렬로 실행할 작업 수 (-1이면 전부 다 사용)
]

# 1.데이터
boston_house_prices = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=None)
x = boston_house_prices.iloc[:, :13]
y = boston_house_prices.iloc[:, 13]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

print(x_train.shape, y_train.shape)

# 2.모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('RandomForestRegressor', RandomForestRegressor())])    
# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())                          
model = RandomizedSearchCV(pipe, parameters, cv=5)                     

# 3.훈련
model.fit(x_train, y_train)

# 4.평가
print("acc : ", model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 parameter : ", model.best_params_)
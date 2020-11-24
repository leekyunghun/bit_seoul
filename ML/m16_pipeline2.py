import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline                                        # pipeline import할 내용
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# 1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:,:4]             # (150, 4)
y = iris.iloc[:, 4]             # (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

parameters = [
    {"svc__C": [1, 10, 100, 1000], "svc__kernel":['linear']},
    {"svc__C": [1, 10, 100, 1000], "svc__kernel":['rbf'], 'svc__gamma':[0.001, 0.0001]},
    {"svc__C": [1, 10, 100, 1000], "svc__kernel":['sigmoid'], 'svc__gamma':[0.001, 0.0001]}
]

# 2.모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC())])     # pipeline으로 scaler와 모델을 묶어주었을때는 parameter들의 이름앞에 모델의 이름을 넣어줘야한다. 
                                                                       # ex) 'malddong', SVC() -> {"malddong__C": [1, 10, 100, 1000], "malddong__kernel":['linear']},
# pipe = make_pipeline(MinMaxScaler(), SVC())                          # cross-validation 할때 validation data들은 scaler 범위에서 빠지므로 과적합이 되지않도록 도와준다.
model = RandomizedSearchCV(pipe, parameters, cv=5)                     # cv = 5 -> croos_validation 5개로 나눈다.

# 3.훈련
model.fit(x_train, y_train)

# 4.평가
print("acc : ", model.score(x_test, y_test))
print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 parameter : ", model.best_params_)

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

parameters = [
    {'XGBRegressor__n_estimators':[100, 200, 300], 'XGBRegressor__learning_rate':[0.1, 0.3, 0.001, 0.01],
     'XGBRegressor__max_depth':[4, 5, 6]},

    {'XGBRegressor__n_estimators':[90, 100, 110], 'XGBRegressor__learning_rate':[0.1, 0.001, 0.01],
     'XGBRegressor__max_depth':[4, 5, 6], 'XGBRegressor__colsample_bytree':[0.6, 0.9, 1]},

     {'XGBRegressor__n_estimators':[90, 110], 'XGBRegressor__learning_rate':[0.1, 0.001, 0.5],
     'XGBRegressor__max_depth':[4, 5, 6], 'XGBRegressor__colsample_bytree':[0.6, 0.9, 1],
     'XGBRegressor__colsample_bylevel':[0.6, 0.7, 0.9]}
]

n_jobs = -1

# 1. 데이터
boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size = 0.8, shuffle = True, random_state = 66)

# 2.모델
pipe = Pipeline([('scaler', RobustScaler()), ('XGBRegressor', XGBRegressor())])    
# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())                          
model = RandomizedSearchCV(pipe, parameters, cv=5)                     

# 3.훈련
model.fit(x_train, y_train)

# 4.평가, 예측
y_predict = model.predict(x_test)
print("최종정답률 : ", r2_score(y_test, y_predict))

# 최종정답률 :  0.9423380214915009
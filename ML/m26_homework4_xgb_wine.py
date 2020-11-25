# csv 활용
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

parameters = [
    {'XGBClassifier__n_estimators':[100, 200, 300], 'XGBClassifier__learning_rate':[0.1, 0.3, 0.001, 0.01],
     'XGBClassifier__max_depth':[4, 5, 6]},

    {'XGBClassifier__n_estimators':[90, 100, 110], 'XGBClassifier__learning_rate':[0.1, 0.001, 0.01],
     'XGBClassifier__max_depth':[4, 5, 6], 'XGBClassifier__colsample_bytree':[0.6, 0.9, 1]},

     {'XGBClassifier__n_estimators':[90, 110], 'XGBClassifier__learning_rate':[0.1, 0.001, 0.5],
     'XGBClassifier__max_depth':[4, 5, 6], 'XGBClassifier__colsample_bytree':[0.6, 0.9, 1],
     'XGBClassifier__colsample_bylevel':[0.6, 0.7, 0.9]}
]

n_jobs = -1

# 1. 데이터
wine = pd.read_csv("./data/csv/winequality-white.csv", header = 0, index_col = None, sep = ';')
wine = wine.values

x = wine[:, :-1]
y = wine[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

# 2.모델
pipe = Pipeline([('scaler', MinMaxScaler()), ('XGBClassifier', XGBClassifier())])    
# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())                          
model = RandomizedSearchCV(pipe, parameters, cv=4)                     

# 3.훈련
model.fit(x_train, y_train)

# 4.평가, 예측
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("acc : ", acc)

# acc :  0.686734693877551
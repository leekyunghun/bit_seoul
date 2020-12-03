# 실습
# 상단 모델에 그리드서치 또는 랜덤서치 학습
# 최적의 R2값과 feature_importances를 구할것

# 위 쓰레드값으로 SelectFromModel를 구해서
# 최적의 feature 갯수를 구할 것

# 위 feature 갯수로 데이터를 수정해서 
# 그리드서치 또는 랜덤서치 적용해서
# 최적의 R2값을 구할 것 

# 1번값과 2번값을 비교할 것

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

parameter = [
             {'base_score': [0.5], 'booster':['gbtree'], 'colsample_bylevel':[0.7],
             'colsample_bynode':[1], 'colsample_bytree':[1], 'gamma':[0], 'gpu_id':[-1],
             'importance_type':['gain'], 'interaction_constraints':[''],
             'learning_rate':[0.1], 'max_delta_step':[0], 'max_depth':[5],
             'min_child_weight':[1], 'missing':[None], 'monotone_constraints':['()'],
             'n_estimators':[90], 'n_jobs':[0], 'num_parallel_tree':[1], 'random_state':[0],
             'reg_alpha':[0], 'reg_lambda':[1], 'scale_pos_weight':[1], 'subsample':[1],
             'tree_method':['exact'], 'validate_parameters':[1], 'verbosity':[None]}
]


n_jobs = -1
x, y = load_boston(return_X_y=True)
x = np.delete(x, [1,3,11], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

model = RandomizedSearchCV(XGBRegressor(), parameter, cv = 5)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("R2 : ", score)

print(model.best_estimator_.feature_importances_)
thresholds = np.sort(model.best_estimator_.feature_importances_)
print(model.best_estimator_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    
    selection_model = RandomizedSearchCV(XGBRegressor(), parameter, cv = 5)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)

    print("Thresh = %.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score * 100.0))

# RandomSearch R2값
# R2 :  0.9273120717462751

# 최적의 파라미터
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=5,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=90, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

# 최적의 feature 갯수
# Thresh = 0.011, n = 11, R2: 94.25%

# feature 갯수 수정하여 뽑은 R2
# R2 :  0.9360477431160328
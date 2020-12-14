import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

samsung_x = np.load("./Samsung/samsung_x.npy", allow_pickle=True).astype('float32')
samsung_y = np.load("./Samsung/samsung_y.npy", allow_pickle=True).astype('float32')
samsung_predict = np.load("./Samsung/samsung_predict.npy", allow_pickle=True).astype('float32')
samsung_y = samsung_y.reshape(659, 1)
samsung_predict = samsung_predict.reshape(1, samsung_predict.shape[0])

bit_computer_x = np.load("./Samsung/bit_computer_x.npy", allow_pickle=True).astype('float32')
bit_computer_predict = np.load("./Samsung/bit_computer_predict.npy", allow_pickle=True).astype('float32')
bit_computer_predict = bit_computer_predict.reshape(1, bit_computer_predict.shape[0])

gold_x = np.load("./Samsung/gold_x.npy", allow_pickle=True).astype('float32')
gold_predict = np.load("./Samsung/gold_predict.npy", allow_pickle=True).astype('float32')
gold_predict = gold_predict.reshape(1, gold_predict.shape[0])

kosdaq_x = np.load("./Samsung/kosdaq_x.npy", allow_pickle=True).astype('float32')
kosdaq_predict = np.load("./Samsung/kosdaq_predict.npy", allow_pickle=True).astype('float32')
kosdaq_predict = kosdaq_predict.reshape(1, kosdaq_predict.shape[0])

# print(samsung_x.shape, samsung_y.shape, samsung_predict.shape)   
# print(bit_computer_x.shape, bit_computer_predict.shape)         
# print(gold_x.shape, gold_predict.shape)                         
# print(kosdaq_x.shape, kosdaq_predict.shape)               

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

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size = 0.7, shuffle = True, random_state = 66)
bit_computer_x_train, bit_computer_x_test = train_test_split(bit_computer_x, train_size = 0.7, shuffle = True, random_state = 66)
gold_x_train, gold_x_test = train_test_split(gold_x, train_size = 0.7, shuffle = True, random_state = 66)
kosdaq_x_train, kosdaq_x_test = train_test_split(kosdaq_x, train_size = 0.7, shuffle = True, random_state = 66)


# 2.모델
# pipe = Pipeline([('scaler', MinMaxScaler()), ('XGBRegressor', XGBRegressor())])    
# # pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())                          
# model = RandomizedSearchCV(pipe, parameters, cv=4) 
model = XGBRegressor(base_score=0.5, booster='gbtree',
                              colsample_bylevel=1, colsample_bynode=1,
                              colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain',
                              interaction_constraints='', learning_rate=0.1,
                              max_delta_step=0, max_depth=6, min_child_weight=1,
                              monotone_constraints='()',
                              n_estimators=300, n_jobs=0, num_parallel_tree=1,
                              random_state=0, reg_alpha=0, reg_lambda=1,
                              scale_pos_weight=1, subsample=1,
                              tree_method='exact', validate_parameters=1,
                              verbosity=None)
# 3.훈련
model.fit(samsung_x_train, samsung_y_train, verbose=1)

# 4.평가, 예측
y_predict = model.predict(samsung_predict)
print("XGBoost : ", y_predict)

# XGBoost :  [63896.434]
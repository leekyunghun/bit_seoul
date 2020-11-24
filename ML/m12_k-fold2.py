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

model1 = LinearSVC()
model2 = SVC()
model3 = KNeighborsClassifier()
model4 = KNeighborsRegressor()
model5 = RandomForestClassifier()
model6 = RandomForestRegressor()

scores1 = cross_val_score(model1, x_train, y_train, cv = kfold)       # cross_validation에서 단계마다 나오는 scroe
scores2 = cross_val_score(model2, x_train, y_train, cv = kfold)     
scores3 = cross_val_score(model3, x_train, y_train, cv = kfold)    
scores4 = cross_val_score(model4, x_train, y_train, cv = kfold)      
scores5 = cross_val_score(model5, x_train, y_train, cv = kfold)    
scores6 = cross_val_score(model6, x_train, y_train, cv = kfold)      

print('\nLinearSVC_scores : ', scores1)
print('\nSVC_scores : ', scores2)
print('\nKNeighborsClassifier_scores : ', scores3)
print('\nKNeighborsRegressor_scores : ', scores4)
print('\nRandomForestClassifier_scores : ', scores5)
print('\nRandomForestRegressor_scores : ', scores6)


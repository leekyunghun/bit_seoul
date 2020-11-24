# 기준 xgboost
# 1. feature importances가 0인 값들을 제거 or 하위 30% 제거
# 디폴트와 성능 비교
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                           # xgboost -> cmd에서 pip install xgboost로 다운

# 1.데이터
wine = pd.read_csv("./data/csv/winequality-white.csv", header = 0, index_col = None, sep = ';')
wine = wine.values

x = wine[:, :-1]
y = wine[:, -1]

x= np.delete(x, [0, 4, 8], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_wine(model):
#     n_features = wine.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), wine.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_wine(model)
# plt.show()

# Default
# acc :  0.6551020408163265
# [0.06218192 0.11186972 0.07586386 0.07901246 0.0632134  0.08535022
#  0.06725179 0.06797317 0.06311163 0.07061544 0.25355643]

# FI
# acc :  0.636734693877551
# [0.14471425 0.09513848 0.10052259 0.10946164 0.09027013 0.09050501
#  0.08920547 0.28018242]
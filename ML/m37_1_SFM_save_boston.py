from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score
import pickle

x, y = load_boston(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)


model = XGBRegressor(n_jobs=-1)
model.fit(x_train, y_train)
predict = model.predict(x_test)
score = model.score(x_test, y_test)
print("R2 : ", score)

# thresholds = np.sort(model.feature_importances_)

# print(thresholds)

# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)

#     select_x_train = selection.transform(x_train)
    
#     selection_model = XGBRegressor(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)

#     pickle.dump(selection_model, open("./model/xgb/SFM_boston/boston.pickle."+ str(select_x_train.shape[1]) + ".dat", "wb"))

#     select_x_test = selection.transform(x_test)
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)

#     print("Thresh = %.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score * 100.0))

model2 = pickle.load(open("./model/xgb/SFM_boston/boston.pickle.12.dat", "rb"))
print("불러왔다!!")

score = model2.score(x_test, y_test)
print("R2 : ", score)
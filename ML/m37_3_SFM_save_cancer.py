from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print("accuracy : ", score)

thresholds = np.sort(model.feature_importances_)

print(thresholds)

# for thresh in thresholds:
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)

#     select_x_train = selection.transform(x_train)
    
#     selection_model = XGBClassifier(n_jobs=-1)
#     selection_model.fit(select_x_train, y_train)

#     model.save_model("./model/xgb/SFM_cancer/cancer.xgb."+ str(select_x_train.shape[1]) + ".model")
#     print("저장완료!!")
    
#     select_x_test = selection.transform(x_test)
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)

#     print("Thresh = %.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score * 100.0))

model2 = XGBClassifier()
model2.load_model("./model/xgb/SFM_cancer/cancer.xgb.30.model")
print("불러왔다!!")

acc2 = model2.score(x_test, y_test)
print("acc2 : ", acc2)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                          

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
# model = GradientBoostingClassifier(max_depth=4)
model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

########################### xgboost 모델 저장, 불러오기 ################################

# import pickle                                    
# pickle.dump(model, open("./model/xgb/cancer.pickle.dat", "wb"))

# import joblib
# joblib.dump(model, "./model/xgb/cancer.joblib.dat")

# model.save_model("./model/xgb/cancer.xgb.model")
# print("저장완료!!")

# model2 = pickle.load(open("./model/xgb/cancer.pickle.dat", "rb"))
# model2 = joblib.load("./model/xgb/cancer.joblib.dat")

model2 = XGBClassifier()
model2.load_model("./model/xgb/cancer.xgb.model")
print("불러왔다!!")

acc2 = model2.score(x_test, y_test)
print("acc2 : ", acc2)
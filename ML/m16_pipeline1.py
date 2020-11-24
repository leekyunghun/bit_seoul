import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline                                    # pipeline import할 내용
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# 1.데이터
iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:,:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

pipe = make_pipeline(MinMaxScaler(), SVC())     # cross-validation 할때 validation data들은 scaler 범위에서 빠지므로 과적합이 되지않도록 도와준다.
pipe.fit(x_train, y_train)

print("acc : ", pipe.score(x_test, y_test))


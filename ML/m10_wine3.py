import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('./data/csv/winequality-white.csv', sep = ';', header = 0)

count_data = wine.groupby('quality')['quality'].count()        # 해당 열에 있는 각 개체들을 카운트해줌

print(count_data)

# quality
# 3      20
# 4     163
# 5    1457
# 6    2198
# 7     880
# 8     175
# 9       5

count_data.plot()
plt.show()
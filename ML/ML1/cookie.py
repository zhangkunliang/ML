import pandas as pd
from pandas._libs.reshape import explode
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import matplotlib

data = pd.read_excel("data.xlsx")

# 加载数据集
X = data[["反应温度", "离心转速", "超声频率（W）", "超声时间(min)"]]
print(type(X), X.shape)
Y = data[["荧光强度 106"]]
# names = boston["feature_names"]
# print(names)
rf = RandomForestRegressor()
rf.fit(X, Y.values.ravel())
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), X), reverse=True))

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
labels = ["Reaction Temperature", "Centrifugal Speed", "Ultrasonic Frequency", "Ultrasonic Time"]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0.2, 0, 0, 0)
pa = rf.feature_importances_
plt.pie(pa, labels=labels, autopct='%0.2f%%', labeldistance=3, colors=colors, explode=explode, shadow=True,
        startangle=45, textprops={'fontsize': 13, 'color': 'k'}, radius=1.4)

plt.legend(loc='center left', prop="5")
plt.savefig('D:\ProgramFiles\PycharmProject\pythonProject\ML1\picture\cookie.jpg')
plt.show()

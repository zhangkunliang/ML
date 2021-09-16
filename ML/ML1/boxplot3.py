import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 5))#设置画布
data = pd.read_excel("aep1.xlsx")#读取表格数据
#ax1=fig.add_subplot(1, 1, 1)#编号布局
#ax1.scatter([-0.2] * len(rf_train_aep), rf_train_aep, color='red', marker='.')#描点
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Algorithm set", y="Absolute Percentage error(%)", hue="Split", data=data)
fig.savefig('picture/boxplot.png')
plt.show()

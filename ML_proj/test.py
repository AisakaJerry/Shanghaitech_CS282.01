import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()  # 导入数据集


# 数据转化函数
def skdata2df(skdata):
    dfdata = pd.DataFrame(skdata.data, columns=skdata.feature_names)
    dfdata["target"] = skdata.target
    return dfdata


bs = skdata2df(boston)
print(bs)

boston_X = bs[["RM", "LSTAT", 'PTRATIO']]  # 参数
# boston_X = boston.data
print(boston_X)

boston_y = boston.target  # 结果

X_train, X_test, y_train, y_test = train_test_split(
    boston_X, boston_y, test_size=0.3)  # 划分测试集合训练集

linreg = LinearRegression(normalize=True)  # 线性回归模型训练（最小二乘法）

linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(y_pred)
print(y_test)
print(linreg.coef_)  # 斜率
print(linreg.intercept_)  # 截距
print(linreg.score(X_test, y_test))  # 评分

x = np.linspace(1, 50, 50)  # 生成序列 (用于对比)
y = x  # (用于对比)

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="red")  # 测试集结果与训练结果分别为xy
ax.plot(x, y, 'k--')  # 对比曲线
ax.set_xlabel('y_test')
ax.set_ylabel('y_pred')
plt.show()

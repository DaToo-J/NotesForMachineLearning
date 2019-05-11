import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 加载数据集
dataset = pd.read_csv('../datasets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

# 切分数据集
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

# 训练模型
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# 预测结果
Y_pred = regressor.predict(X_test)

# 可视化
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.plot(X_test , Y_pred, color ='green')
plt.show()

# 求解后的参数
print(X_test)
print(Y_pred)
print(regressor.coef_, regressor.intercept_)

for i in range(7):
    y = X_test[i][0] * regressor.coef_[0] + regressor.intercept_
    print(y, end=', ')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 加载数据集
dataset = pd.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# 将类别数据数字化
# 类别数据：在数据集中没有用数字表示的，可以划分类别的数据，将其分类再用数字表示
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:,3])    # 将第四列（城市）转化为数字表示
onehotencoder = OneHotEncoder(categorical_features=[3]) # 对第四列进行onehot编码
X = onehotencoder.fit_transform(X).toarray()    # 将X用onehot表示，依然需要先fit再transform
# print(X.shape)

# 躲避虚拟变量陷阱
# 虚拟变量陷阱：多个变量之间高度相关，可用一个变量来表示
X = X[:, 1:]

# 拆分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# 训练多元线性回归模型
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# 预测结果
y_pred = regressor.predict(X_test)

# 看一下下参数嘛
print('x test : ',X_test)
print('y pred : ',', '.join([str(i) for i in y_pred]))
print('parameters: ',regressor.coef_, regressor.intercept_)

print('y : ')
for i in range(len(X_test)):
    y = X_test[i][0] * regressor.coef_[0] +  X_test[i][1] * regressor.coef_[1] +  X_test[i][2] * regressor.coef_[2] +  X_test[i][3] * regressor.coef_[3] +  X_test[i][4] * regressor.coef_[4] + regressor.intercept_
    print(y, end=', ')
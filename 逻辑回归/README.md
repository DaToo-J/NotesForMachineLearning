
# 四、逻辑回归 VS 线性回归

## 4.1 **步骤**
1. 加载数据集
2. 数据预处理 
    - `from sklearn.preprocessing import package`
    - **类别数据数字化。** 类别数据：在数据集中没有用数字表示的，可以划分类别的数据，将其分类再用数字表示
        - `LabelEncoder(), OneHotEncoder()`
    - **躲避虚拟变量陷阱。** 虚拟变量陷阱：多个变量之间高度相关，可用一个变量来表示
    - **特征转化。** 为了让数据更好的被表示，见下文。
3. 拆分数据集。`train_test_split(X,Y, test_size=0.2, random_state=0)`
4. 训练线性回归模型。
    - `LinearRegression().fit(X_train, Y_train)`
    - `LogisticRegression().fit(X_train, y_train)`
5. 预测结果。`regressor.predict(X_test)`
6. 评估模型。
    - **混淆模型。** 可用于评价分类模型，见下文。
7. 可视化。
    



## 4.2 **sklearn 彩蛋**

1. `fit()`：调用api之前需要 fit适配 一下，并不是 train训练
    - 求得训练集 X 的固有属性：均值、方差、min、max
2. `transform()`：在`fit`之后，进行标准化、降维、归一化等操作
3. `fit_transform()`：组合前两者，对数据进行某种统一处理。
    - 先对部分数据进行fit拟合，找到该part数据的整体指标，然后对该part数据进行transform转换；
    - 再根据上一步的整体指标，对剩余数据进行transform；
    - 从而保证train、test处理方式相同


## 4.3 **特征工程**
*是为了要让数据成为有效的特征*

1. 常用方法：标准化、归一化、特征离散化、正则化
[阿里云-特征工程](https://yq.aliyun.com/articles/577701)
2. StandardScaler: 特征转化，标准化。转化为均值为零、方差为一的正态分布
    - **Z-score标准化** ： 标准化数据 = (原数据 - 均值) / 标准差





## 4.4 **评价模型**

1. 分类型模型的评价指标有：
    - 混淆矩阵，also 误差矩阵
    - ROC曲线
    - AUC面积

2. **混淆矩阵及相关**
    - 分别统计观测正确、观测错误的个数，并将其放在一个表中表示。
    - ![20180531113257203.png-4.4kB][2]


| 指标 | 指标 |
|---|----|
| 混淆矩阵 | TP & TN越大，FP & FN越小，越好
| 二级指标 | ![20180531115939413.png-26.8kB][3]
| 三级指标 | ![image_1dagv948kcgakd2rb0ldlrus11.png-24.2kB][4]
|| [混淆矩阵-有实例](https://blog.csdn.net/Orange_Spotty_Cat/article/details/80520839) 

  [2]: http://static.zybuluo.com/HelloDatoo/sqcl9gdwbsvqmzztb80nw5g5/20180531113257203.png
  [3]: http://static.zybuluo.com/HelloDatoo/vs7orubnkqbbmubmv5h5wclo/20180531115939413.png
  [4]: http://static.zybuluo.com/HelloDatoo/k2tqyimhkvq0qu12og8554y2/image_1dagv948kcgakd2rb0ldlrus11.png

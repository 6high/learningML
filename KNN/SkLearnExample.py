# -*- coding: utf-8 -*-
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

# 导入数据集
iris = datasets.load_iris()


print iris

# knn训练
knn.fit(iris.data, iris.target)

# 预测
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print '\n\n\n',predictedLabel
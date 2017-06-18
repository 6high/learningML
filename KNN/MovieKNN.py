# -*- coding: utf-8 -*-

import numpy as np
from sklearn import neighbors

# kkn算法库
knn = neighbors.KNeighborsClassifier()

# 电影分类
data = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])

labels = np.array([1,1,1,2,2,2])

# 训练
knn.fit(data,labels)

# 预测
predictedLabel = knn.predict([18,90])

print predictedLabel
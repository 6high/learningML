# -*- coding: utf-8 -*-

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing

# Read in the csv file and put features into list of dict and list of class label
allElectornicsData = open('data.csv', 'r')
reader = csv.reader(allElectornicsData)
headers = reader.next()    # python2.7 supported   本质获取csv 文件的第一行数据
#headers = reader.__next__()    python 3.5.2
headers = next(reader)

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)
print(labelList)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummyX: " + str(dummyX))
print(vec.get_feature_names())
print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: ", str(dummyY))

# Using decision tree for classification        ===========【此处调用为算法核心】============
#clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(dummyX, dummyY)
print("clf: ", str(clf))

# Visualize model
# dot -Tpdf iris.dot -o ouput.pdf
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names = vec.get_feature_names(), out_file = f)


# predict
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY: " + str(predictedY))
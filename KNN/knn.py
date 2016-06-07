'''
Created on Dec 13, 2015

@author: lqy
'''
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier() # define a knn classification

iris = datasets.load_iris() # load iris dataset

#print iris.get('data')

knn.fit(iris.data, iris.target)

predictLable = knn.predict([[0.1,0.4,3.9,2.3]])

print "predictResult: ", predictLable
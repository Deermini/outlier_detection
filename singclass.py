# !usr/bin/env python
# -*- coding:utf-8 -*-

from sklearn import svm
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

positive=np.load("/home/deermini/PycharmProjects/medical-analysis/医保数据集/data/train_positive.npy")
negative=np.load("/home/deermini/PycharmProjects/medical-analysis/医保数据集/data/train_negative.npy")
print(positive.shape)
print(negative.shape)

positive_trainX,positive_testX,positive_trainy,positive_testy=train_test_split(positive[:,:9],positive[:,9], test_size=0.1,random_state=33)
negative_trainX,negative_testX,negative_trainy,negative_testy=train_test_split(negative[:,:9],negative[:,9], test_size=0.1,random_state=33)

clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(positive_trainX)

#对存在盗窃的人进行预测
prediction=clf.predict(positive_testX)
count=list(prediction).count(1)
acc=1.0*count/len(prediction)
print("positive_testy:",acc)

#对正常人群进行预测
prediction2=clf.predict(negative_trainX)
count=list(prediction2).count(-1)
acc2=1.0*count/len(prediction2)
print("negative_trainX:",acc2)

#############################################################

clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(negative_trainX)

prediction=clf2.predict(negative_testX)
count=list(prediction).count(1)
acc=1.0*count/len(prediction)
print("negative_testX:",acc)

prediction2=clf2.predict(positive_trainX)
count=list(prediction2).count(-1)
acc2=1.0*count/len(prediction2)
print("positive_trainX:",acc2)




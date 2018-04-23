#!usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

def figure_play(x,pos_y,nag_y):
    """绘制缺失值的比例图"""
    plt.figure(21)
    plt.subplot(211)
    plt.plot(x,pos_y,c="red")
    plt.subplot(212)
    plt.plot(x,nag_y,c="blue")
    plt.show()

def figure_object(pos_data,nag_data,li):
    """绘制类别型变量图"""
    plt.figure(21)
    plt.subplot(211)
    data_dict=Counter(pos_data)
    # print("pos_data_dict.keys:",list(data_dict.keys()))
    # print("pos_data_dict.vales:", list(data_dict.values()))
    plt.plot(range(len(list(data_dict.keys()))),list(data_dict.values()), c="red")
    plt.subplot(212)
    data_dict = Counter(nag_data)
    plt.plot(range(len(list(data_dict.keys()))),list(data_dict.values()), c="blue")
    plt.xlabel("the figure of %s"%li)
    plt.show()


def Missing_stat(pos_data,nag_data):
    """统计各列的缺失值"""
    columns=pos_data.columns
    pos_miss_count=[];nag_miss_count=[]
    pos_length=pos_data.shape[0];nag_length=nag_data.shape[0]
    for li in columns:
        count=sum(pd.isnull(pos_data[li]))
        pos_miss_count.append(1.0*count/pos_length)
        count = sum(pd.isnull(nag_data[li]))
        nag_miss_count.append(1.0 * count / nag_length)
    #figure_play(range(len(columns)),pos_miss_count,nag_miss_count)
    miss_above_half_index=[columns[i] for i in range(len(columns)) if pos_miss_count[i]>0.5 and nag_miss_count[i]>0.5 ]
    print("the count of miss_above_half:",len(miss_above_half_index))
    return miss_above_half_index

def load_data():
    """read data"""
    root="F:/projectfile/yibao_yishi_data/yibao_Feature_Engineering/data/"
    pos_data=pd.read_csv(os.path.join(root,"outlier/mzsf.csv"),encoding="cp936")
    nag_data=pd.read_csv(os.path.join(root,"random_5K/mzsf_5k.csv"),encoding="cp936")
    del pos_data['XMING1']

    print(pos_data.shape)
    print(nag_data.shape)

    # print(pos_data.GZZT00.astype(str).describe())
    # print(nag_data.GZZT00.dtypes)

    miss_above_half_index=Missing_stat(pos_data,nag_data)
    """去除缺失值超过一半的列"""
    for li in miss_above_half_index:
        del pos_data[li]
        del nag_data[li]

    """处理缺失值"""
    pos_data=pos_data.dropna()
    nag_data=nag_data.dropna()
    print("pos_data.dropna:",pos_data.shape)
    print("nag_data.dropna:",nag_data.shape)
    #print(pos_data.columns)

    """转换数据类型"""
    for li in pos_data.columns:
        if pos_data[li].dtypes=="int64":
            pos_data[li]=pos_data[li].astype(str)
            nag_data[li]=nag_data[li].astype(str)

    """将清洗后的数据存入文件"""
    pos_data.to_csv(os.path.join(root,"pos_data.csv"),index=False)
    nag_data.to_csv(os.path.join(root,"nag_data.csv"),index=False)


def machion_learning():
    """利用机器学习算法来进行处理"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,recall_score,precision_score
    from sklearn.model_selection import train_test_split

    root="F:/projectfile/yibao_yishi_data/yibao_Feature_Engineering/data/"
    pos_data=pd.read_csv(os.path.join(root,"pos_data.csv"),encoding="cp936")
    nag_data=pd.read_csv(os.path.join(root,"nag_data.csv"),encoding="cp936")
    print("pos_data:", pos_data.shape)
    print("nag_data:", nag_data.shape)
    print(pos_data.GHKSMC.dtypes)
    index=[]
    for li in pos_data.columns:
        if pos_data[li].dtypes=="float64":
            index.append(li)
        else:
            figure_object(pos_data[li],nag_data[li],li)


    print("the count of index:",len(index))
    x=np.vstack((pos_data.ix[:,index].values,nag_data.ix[:,index].values))
    y=list(np.ones(pos_data.shape[0]))+list(np.zeros(nag_data.shape[0]))

    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.6,random_state=33)
    clf=RandomForestClassifier()
    #clf =LogisticRegression()
    clf.fit(train_x,train_y)
    pred=clf.predict(test_x)
    acc=accuracy_score(test_y,pred)
    recall=recall_score(test_y,pred)
    precision=precision_score(test_y,pred)
    print("acc:",acc)
    print("recall:",recall)
    print("precision:",precision)

#load_data()
machion_learning()

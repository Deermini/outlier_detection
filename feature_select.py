#!usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.model_selection import train_test_split

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
    """read data 两张表分别处理"""
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

def load_data_combine():
    """把两张表合起来进行处理"""
    root = "F:/projectfile/yibao_yishi_data/yibao_Feature_Engineering/data/"
    pos_data = pd.read_csv(os.path.join(root, "outlier/mzsf.csv"), encoding="cp936")
    nag_data = pd.read_csv(os.path.join(root, "random_5K/mzsf_5k.csv"), encoding="cp936")
    del pos_data['XMING1']
    pos_data["label"]=pd.Series(1,index=pos_data.index)
    nag_data['label']=pd.Series(0,index=pos_data.index)

    print(pos_data.shape)
    print(nag_data.shape)
    #print(pos_data['label'].dtypes)

    combine_data=pd.DataFrame(pd.concat([pos_data,nag_data],axis=0,ignore_index=True))
    print(combine_data.shape)

    miss_above_half_index=Missing_stat(pos_data,nag_data)
    """去除缺失值超过一半的列"""
    for li in miss_above_half_index:
        del combine_data[li]

    """处理缺失值"""
    combine_data=combine_data.dropna()
    print("combine_data.dropna:",combine_data.shape)
    #print("the number of columns",len(combine_data.columns))

    """转换数据类型"""
    combine_data_str=[]
    for li in combine_data.columns:
        if combine_data[li].dtypes=="int64":
            combine_data[li]=combine_data[li].astype(str)

    """去除一些没有用的列信息"""
    delect_index = ["GRID00","DJLSH0","MZLSH0","DWID01", "GHKSMC", "FWWDBH", "SJSFRQ", "SJSFSJ", "SFRQ00", "SFSJ00", "BQZDMS","QSRQ00"]
    for li in delect_index:
        del combine_data[li]

    print(pos_data.shape)
    print(nag_data.shape)

    columns_index=[]
    for li in combine_data.columns:
        if combine_data[li].dtypes!='float64':
            content = list(set(list((combine_data[li]))))
            length=len(content)
            if length>=2 and length<=2000:
                print(combine_data[li].dtypes, li,len(content),content)
                columns_index.append(li)

    print("columns_index:", columns_index)
    print("the number of columns_index:", len(columns_index))

    """将处理好的columns_index和float64进行分析"""
    index=[]
    for li in combine_data.columns:
        type=combine_data[li].dtypes
        if type=='float64' or li in columns_index:
            index.append(li)
    #index.append("label")

    print("index:",index)
    print("the number of index:", len(index))
    combine_data=combine_data.ix[:,index]

    """将清洗后的数据存入文件"""
    combine_data.to_csv(os.path.join(root,"combine_data.csv"),index=False)


def machion_learning():
    """利用机器学习算法来进行处理"""
    root="F:/projectfile/yibao_yishi_data/yibao_Feature_Engineering/data/"
    def load(root=root):
        pos_data=pd.read_csv(os.path.join(root,"pos_data.csv"),encoding="cp936")
        nag_data=pd.read_csv(os.path.join(root,"nag_data.csv"),encoding="cp936")
        print("pos_data:", pos_data.shape)
        print("nag_data:", nag_data.shape)
        print(pos_data.GHKSMC.dtypes)
        index=[]
        for li in pos_data.columns:
            if pos_data[li].dtypes=="float64":
                index.append(li)
            # else:
            #     figure_object(pos_data[li],nag_data[li],li)

        print("the count of index:",len(index))
        x=np.vstack((pos_data.ix[:,index].values,nag_data.ix[:,index].values))
        y=list(np.ones(pos_data.shape[0]))+list(np.zeros(nag_data.shape[0]))
        return x,y

    def load_combine(root=root):
        combine_data = pd.read_csv(os.path.join(root, "combine_data.csv"), encoding="cp936")
        y=combine_data["label"].values
        print("Counter(y):",Counter(y))
        del combine_data["label"]
        del combine_data["SFFS00"]  #
        del combine_data["DWLB00"]  #单位类别
        del combine_data["ICZTBH"]  #IC卡状态

        """类别解码"""
        for li in combine_data.columns:
            if combine_data[li].dtypes!="float64":
                class_mapping={label:idx for idx,label in enumerate(np.unique(combine_data[li]))}
                combine_data[li]=combine_data[li].map(class_mapping)
        x=combine_data.values
        return x,y,combine_data.columns

    x,y,columns_index=load_combine(root)
    #x, y = load(root)
    print("x.shape",x.shape)
    read_bin(root,columns_index)
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=33)
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

def read_bin(root,index_columns):
    ranking = np.fromfile(root + 'ranking.bin', dtype='int32')
    print(ranking)
    permutation = ranking.argsort()
    f=open(root+"ranking.txt","w")
    for i, ind in enumerate(permutation):
        print(i, index_columns[ind])
        f.write(str(i)+":"+index_columns[ind]+"\n")

def get_ranking_index(root):
    f = open(root + "ranking.txt")
    ranking_index=[]
    for li in f.readlines():
        li=li.strip().split(":")[1]
        ranking_index.append(li)
    return ranking_index

def check_test(index=None):
    """把两张表合起来进行处理"""
    root = "F:/projectfile/yibao_yishi_data/yibao_Feature_Engineering/data/"
    pos_data = pd.read_csv(os.path.join(root, "outlier/mzsf.csv"), encoding="cp936")
    nag_data = pd.read_csv(os.path.join(root, "random_5K/mzsf_5k.csv"), encoding="cp936")
    del pos_data['XMING1']
    pos_data["label"] = pd.Series(1, index=pos_data.index)
    nag_data['label'] = pd.Series(0, index=pos_data.index)

    print(pos_data.shape)
    print(nag_data.shape)
    # print(pos_data['label'].dtypes)

    combine_data = pd.DataFrame(pd.concat([pos_data, nag_data], axis=0, ignore_index=True))
    print(combine_data.shape)

    """抽取相应的列进行检查"""'ZHZFE0,GRZFE0,JJZFE0'
    ranked_index = get_ranking_index(root)[:30]
    #index=['LJTJJ1',"LJTXJ1","LJJTC0","LJFYB0","label"]
    index=ranked_index+["label"]

    data=combine_data[index].dropna()
    print(data.shape)

    """类别解码"""
    for li in data.columns:
        if data[li].dtypes != "float64":
            class_mapping = {label: idx for idx, label in enumerate(np.unique(data[li]))}
            data[li] = data[li].map(class_mapping)

    x, y= data.ix[:,index[:-1]].values,data.ix[:,index[-1]].values
    #x=x.reshape([None,1])
    print("x.shape,y.shape", x.shape,y.shape)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=33)
    clf = RandomForestClassifier()
    #clf =LogisticRegression()
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    acc = accuracy_score(test_y, pred)
    recall = recall_score(test_y, pred)
    precision = precision_score(test_y, pred)
    print("acc:", acc)
    print("recall:", recall)
    print("precision:", precision)


#load_data()
load_data_combine()
machion_learning()
check_test()




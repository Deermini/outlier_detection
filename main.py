# !usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import csv
import codecs
from numpy import *
import Apriori.apriori
from sklearn.externals import joblib


class medical_solution_to_time():
    """获取每个人就医的时间情况"""
    def train(self):
        #data=pd.read_csv("F:/project_files/yibao_data/random_5k/mzsf_5k.csv")
        data=pd.read_csv("F:/project_files/yibao_data/outlier/mzsf.csv")
        #load_data=data.set_index("SJSFRQ")
        train_data=data.ix[:,['GRID00','SJSFRQ']].values

        mz_dict={}
        for li,lj in train_data:
            li=li.strip()
            if li not in mz_dict.keys():
                mz_dict[li]=[]
            mz_dict[li].append(int(lj))

        yc_data=[]
        for key in mz_dict.keys():
            content=[li for li in list(set(mz_dict[key])) if li >20160630 and li<20170701]
            yc_data.append(content)

        for li in yc_data:
            print(len(sorted(li)))
            print(sorted(li))

# medic=medical_solution_to_time()
# medic.train()

class medical_solution_to_people():
    """获取每个时间段的就医人数情况"""
    def train(self):
        data=pd.read_csv("F:/project_files/yibao_data/random_5k/mzsf_5k.csv")
        #data2=pd.read_csv("F:/project_files/yibao_data/outlier/mzsf.csv")
        train_data=data.ix[:,['GRID00','SJSFRQ']].values

        people_dict={}
        for li,lj in train_data:
            li=li.strip()
            lj=int(lj)
            if lj>20160630 and lj<20170701:
                if lj not in people_dict.keys():
                    people_dict[lj]=[]
                people_dict[lj].append(li)

        for key in people_dict.keys():
            people_dict[key]=list(set(people_dict[key]))

        sorted_people_dict=sorted(people_dict.items(),key=lambda x:x[0])

        # #打印出字典中存储的情况
        # for key in people_dict.keys():
        #     print(key,len(people_dict[key]))
        #     print(people_dict[key])

        #以某个时间段对数据进行切分（space为日期间隔，2表示把2天并为一天进行计算）
        apriori_data=[]
        start=sorted_people_dict[0][0]
        space=1;space_people=[]
        for key,values in sorted_people_dict:
            if key<(start+space):
                space_people.extend(people_dict[key])
            else:
                apriori_data.append(sorted(space_people))
                space_people=[];start=key+space
                space_people.extend(people_dict[key])

        #利用关联分析对数进行分析
        print("start:")
        L, suppData = Apriori.apriori.apriori(apriori_data, minSupport=0.5)
        rules = Apriori.apriori.generateRules(L, suppData, minConf=0.70)
        print "rules:", rules

# medical=medical_solution_to_people()
# medical.train()


class Lstmclassifier():

    def train(self):
        print("start run")


def load_data():
    def get_data(file=None,file2=None,dict=None):
        """如何更好的获得数据用于LSTM使用"""
        positive_data = {}
        positive = file.ix[:, 'GRID00'].values
        GRID = list(set(positive))

        for li in GRID:
            positive_data[li] = {}

        for lj in file2:
            idx = lj[0]
            idy = [1,3, 4, 5, 6, 7, 8]
            positive_data[idx][lj[2]]=[]
            for id in idy:
                if isinstance(lj[id],str):
                    lj[id]=lj[id].strip()
                    positive_data[idx][lj[2]].append(lj[id])
                else:
                    positive_data[idx][lj[2]].append(lj[id])
                #positive_data[idx][lj[2]].append(lj[id])

        sorted_positive_data=[]
        for key in positive_data.keys():
            count_content = []
            sorted_data=sorted(positive_data[key].items(),key=lambda x:x[0])
            for i,li in enumerate(sorted_data):
                if i!=0 and i%3==0:
                    if count_content!=[]:
                        sorted_positive_data.append(count_content)
                    count_content = []
                    count_content.extend(li[1])
                else:
                    count_content.extend(li[1])
            if count_content != []:
                sorted_positive_data.append(count_content)
        return sorted_positive_data

    data = pd.read_csv("F:/project_files/yibao_data/outlier/mzsf.csv")
    data2 = pd.read_csv("F:/project_files/yibao_data/random_5k/mzsf_5k.csv")
    true_data = data.ix[:,['GRID00', 'SJSFRQ', 'DJLSH0','BQZDBM', 'ZHZFE0', 'GRZFE0', 'JJZFE0', 'BCBXF0','GRZHYE']].dropna()
    false_data=data2.ix[:,['GRID00', 'SJSFRQ', 'DJLSH0','BQZDBM', 'ZHZFE0', 'GRZFE0', 'JJZFE0', 'BCBXF0','GRZHYE']].dropna()
    true_data=true_data.values
    false_data=false_data.values

    BQ_idx=set(data.ix[:,'BQZDBM'].dropna().values)
    BQ_idy=set(data2.ix[:,'BQZDBM'].dropna().values)
    BQ_ID={};i=0
    for li in BQ_idx:
        li=li.strip()
        if li not in BQ_ID.keys():
            BQ_ID[li]=i
            i+=1
    for li in BQ_idy:
        li=li.strip()
        if li not in BQ_ID.keys():
            BQ_ID[li]=i
            i+=1
    print(len(BQ_ID))

    train_x=np.array(get_data(file=data,file2=true_data,dict=BQ_ID))
    train_y=np.array(get_data(file=data2,file2=false_data,dict=BQ_ID))
    print(train_x.shape)
    print(train_y.shape)


    result_data_positive=[]
    for li in train_x:
        if len(li)<20:
            continue
        else:
            code=[]
            #第一条数据
            #data_time=0;code.append(data_time)
            # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[1]]]=1
            # code.extend(BQBM_code)
            a=li[2]/li[5];b=li[3]/li[5];c=li[4]/li[5]
            code.append(a);code.append(b);code.append(c)

            # 第二条数据
            # data_time = np.zeros(100);days=int(li[7])-int(li[0])
            # if  days>299:
            #     days=299
            # data_time[days/3] = 1;code.extend(data_time)
            # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[8]]] = 1
            # code.extend(BQBM_code)
            a = li[9]/li[12];b = li[10]/li[12];c =li[11]/li[12]
            code.append(a);code.append(b);code.append(c)


            # 第三条数据
            # data_time = np.zeros(100);days=int(li[14])-int(li[7])
            # if  days>299:
            #     days=299
            # data_time[days/3] = 1;code.extend(data_time)

            # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[15]]] = 1
            # code.extend(BQBM_code)
            a = li[16] / li[19];b = li[17] / li[19];c = li[18] / li[19]
            code.append(a);code.append(b);code.append(c)


            #第四条数据
            # a = li[23] / li[26];b = li[24] / li[26];c = li[25] / li[26]
            # code.append(a);code.append(b);code.append(c)
            label=[1];code.extend(label)
            result_data_positive.append(code)

    result_data_positive=np.array(result_data_positive)
    print(result_data_positive.shape)
    np.save("data/train_positive.npy", result_data_positive)

    result_data_positive=[]
    result_data_negative = []
    index=np.random.choice(24898,6000)
    train_y=train_y[index]
    for li in train_y:
        if len(li) < 20:
            continue
        else:
            code = []
            # 第一条数据
            #data_time=np.zeros(100);code.extend(data_time)
            # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[1]]] = 1
            # code.extend(BQBM_code)
            a = li[2] / li[5];b = li[3] / li[5];c = li[4] / li[5]
            code.append(a);code.append(b);code.append(c)

            # 第二条数据
            # data_time = np.zeros(100);days=int(li[7])-int(li[0])
            # if days<0:
            #     print(days)
            # days=abs(days)
            # if  days>299:
            #     days=299
            # data_time[days/3] = 1;code.extend(data_time)
            # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[8]]] = 1
            # code.extend(BQBM_code)
            a = li[9] / li[12];b = li[10] / li[12];c = li[11] / li[12]
            code.append(a);code.append(b);code.append(c)

            # 第三条数据
            # data_time = np.zeros(100);days=int(li[14])-int(li[7])
            # if  days>299:
            #     days=299
            # data_time[days/3] = 1;code.extend(data_time)
            # # BQBM_code = np.zeros(3132);BQBM_code[BQ_ID[li[15]]] = 1
            # code.extend(BQBM_code)
            a = li[16] / li[19];b = li[17] / li[19];c = li[18] / li[19]
            code.append(a);code.append(b);code.append(c)

            #第四条数据
            # a = li[23] / li[26];b = li[24] / li[26];c = li[25] / li[26]
            # code.append(a);code.append(b);code.append(c)
            label =[0];code.extend(label)
            result_data_negative.append(code)

    result_data_negative = np.array(result_data_negative)
    print(result_data_negative.shape)
    np.save("data/train_negative.npy", result_data_negative)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle

#load_data()
positive_data=np.load("data/train_positive.npy")
negative_data=np.load("data/train_negative.npy")
data=np.vstack((positive_data,negative_data))
trainX,validX,trainy,validy=train_test_split(data[:,:9],data[:,9], test_size=0.01, random_state=33)

clf=RandomForestClassifier(n_estimators=100,max_depth=15,random_state=33)
clf.fit(trainX,trainy)
pickle.dump(clf,open("model/Randfoest.pickle.dat","wb"))
clf=pickle.load(open("model/Randfoest.pickle.dat","rb"))

prediction=clf.predict(validX)
acc=accuracy_score(validy,prediction)
print(acc)


#读取存储好的模型进行测试
def predict():
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    import numpy as np
    import pickle

    # 读入数据'ZHZFE0'(个人账户支付), 'GRZFE0'(个人支付金额), 'JJZFE0'(基金支付金额), 'BCBXF0'(本次保费合计)
    X1 = raw_input("Please enter the first record:")
    X2 = raw_input("Please enter the second record:")
    X3 = raw_input("Please enter the three record:")
    testX = []
    li = X1.strip().split(',')
    for i in range(3):
        testX.append(float(li[i]) / float(li[3]))
    li = X2.strip().split(',')
    for i in range(3):
        testX.append(float(li[i]) / float(li[3]))
    li = X1.strip().split(',')
    for i in range(3):
        testX.append(float(li[i]) / float(li[3]))

    clf = pickle.load(open("model/Randfoest.pickle.dat", "rb"))
    # clf=pickle.load(open("model/xgb_model.pickle.dat",'rb'))

    prediction = clf.predict_proba(testX)
    print(u"属于异常的概率:", prediction[0][1])






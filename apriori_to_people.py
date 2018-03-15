# !usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import csv
import codecs
from numpy import *
import Apriori.apriori
from sklearn.externals import joblib

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
        f = open("data/apriori.txt", "w")
        for li in rules:
            for i, lj in enumerate(sorted(list(li[0]))):
                # print "i,lj",i,lj
                if (i + 1) != len(list(li[0])):
                    f.write(str(lj) + ",")
                else:
                    f.write(str(lj))
            f.write("--->")
            for i, lj in enumerate(sorted(list(li[1]))):
                if (i + 1) != len(list(li[1])):
                    f.write(str(lj) + ",")
                else:
                    f.write(str(lj))
            f.write("--->")
            f.write(str(li[2]) + "\n")
        f.close()

medical=medical_solution_to_people()
medical.train()











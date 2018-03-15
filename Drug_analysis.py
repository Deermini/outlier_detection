# !usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import codecs

"""
找出欺诈群体药品分布情况并与正常群体用药的分布情况进行对比，从而对数据进行分析。
"""
data=pd.read_csv("F:/project_files/yibao_data/outlier/mzsf.csv")
data2=pd.read_excel("F:/project_files/yibao_data/outlier/mzsfmx.xlsx")
train_data=data.ix[:,['GRID00','DJLSH0']].values
train_data2=data2.ix[:,['DJLSH0','XMMC00']].values

#盗窃人购买的药品
medical=data2.ix[:,'XMMC00'].values
medical_count_dict={}
for li in medical:
    li=li.strip()
    if li not in medical_count_dict.keys():
        medical_count_dict[li]=0
    medical_count_dict[li]+=1
print("medical_count:",len(medical_count_dict.keys()))

#正常人群购买的药品
normal_data=pd.read_excel("F:/project_files/yibao_data/random_5k/mzsfmx_5k.xlsx")
#print(normal_data.dtypes)
medical=normal_data.ix[:,'XMMC00'].values
normal_medical_count_dict={}

for li in medical:
    li=li.strip()
    if li not in normal_medical_count_dict.keys():
        normal_medical_count_dict[li]=0
    normal_medical_count_dict[li]+=1
print("normal_medical_count_dict:",len(normal_medical_count_dict.keys()))

#盗窃与正常人群购买药品的比例
diff={}
i=0
f=codecs.open("data/not_in.txt","w",encoding='UTF-8')
f.write("not_in_normal_medical:"+"\n")
for key in medical_count_dict.keys():
    false=medical_count_dict[key]*1.0/16381
    if key not in normal_medical_count_dict:
        diff[key]=false
        i+=1
        print(i,"not in",key)
        f.write("%3d:%s"%(i,key)+"\n")
    else:
        normal=normal_medical_count_dict[key]*1.0/76886
        diff[key]=false-normal

sorted=sorted(diff.items(),key=lambda x:x[1],reverse=True)[:50]
f.write("key_value:"+"\n")
for key,value in sorted:
    print("key:%s,value:%s"%(key,value))
    f.write("key:%s,value:%s"%(key,value)+"\n")
f.close()

#获取个人ID与单据流水号对应关系
GRID_to_DJ_dict={}
for li,lj in train_data:
    #li = li.strip();lj = lj.split()
    if li not in GRID_to_DJ_dict.keys():
        GRID_to_DJ_dict[li]=[]
    GRID_to_DJ_dict[li].append(int(lj))

#获取单据流水号与药品的对应关系
DJ_to_medical_dict={}
for li,lj in train_data2:
    #li=li.strip();lj=lj.split()
    if li not in DJ_to_medical_dict:
        DJ_to_medical_dict[li]=[]
    DJ_to_medical_dict[li].append(lj)

#print(DJ_to_medical_dict)

#获取个人与药品的对应关系
GRID_to_medical_dict={}
for key in GRID_to_DJ_dict.keys():
    GRID_to_medical_dict[key]=[]
    for li in GRID_to_DJ_dict[key]:
        GRID_to_medical_dict[key].extend(DJ_to_medical_dict[li])
















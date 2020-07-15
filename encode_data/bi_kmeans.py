# -*- coding: UTF-8 -*-
import init_adultdata
import random
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import copy

def init_dataset(index):
    data=np.transpose(init_adultdata.init_AdultData())[index]
    n=len(data)
    dataset=np.zeros((n,2))
    for i in range(n):
        dataset[i][0]=float(data[i])
        dataset[i][1]=float(data[i])
    '''
    for i in range(3000):
        plt.plot(float(data[i]), float(data[i]), 'o', color='r')
    plt.show()'''
    return data,data,dataset
def dis_of_2vec(vec1,vec2):
    #return sqrt(sum(power(vec1-vec2,2)))
    return np.linalg.norm(vec1-vec2)
def random_random():
    return np.random.randn(1,1000)
def random_random1():
    return 2.5*np.random.randn(1,1000)-10
def random_random2():
    return  np.random.randn(1, 1000) +20
def creat_dataset():
    x3=random_random2()
    y3=random_random2()
    x2 = random_random1()
    y2 = random_random1()
    x1 = random_random()
    y1 = random_random()
    x1 = ([i for item in x1 for i in item])
    y1 = ([i for item in y1 for i in item])
    for item in x2:
        for i in item:
            x1.append(i)
    for item in x3:
        for i in item:
            x1.append(i)

    for item in y2:
        for i in item:
            y1.append(i)
    for item in y3:
        for i in item:
            y1.append(i)

    dataset = []  # 数据集
    for i in range(len(x1)):
        dataset.append([x1[i], y1[i]])
    dataset = np.array(dataset)
    plt.plot(x1,y1,'o',color='c')
    plt.title("Scatterplot")
    plt.show()
    return x1,y1,dataset

def get_CenterofK(dataset,k):
    #print shape(dataset)[0]
    row=shape(dataset)[1]
    #row=2
    init_ofKcent=np.mat(zeros((k,row)))
    for j in range(row):
        minJ=min(dataset[:,j])
        rangeJ=float(max(dataset[:,j])-minJ)
        init_ofKcent[:,j]=np.mat(minJ+rangeJ*random.rand(k,1))
    init_ofKcent=init_ofKcent.tolist()
    return init_ofKcent

def k_means(k):
    x1,y1,dataset=creat_dataset()

    init_ofcenterK = get_CenterofK(dataset, k)
    m=shape(dataset)[0]
    clussterAssment = np.zeros((m, 2))
    find =True
    while find:
        find=False
        for i in range(m):
            minIndex=-1
            mindist=inf
            for j in range(k):
                temp_dist=dis_of_2vec(dataset[i],init_ofcenterK[j])
                if temp_dist<mindist:
                    mindist=temp_dist
                    minIndex=j
            if clussterAssment[i][0]!=minIndex: find=True
            clussterAssment[i][0],clussterAssment[i][1]=minIndex,mindist
        temp_x=[0]*k
        temp_y=[0]*k
        temp_count=[0]*k
        for i in range(m):
            for j in range(k):
                if clussterAssment[i][0]==j:
                    temp_x[j]+=dataset[i][0]
                    temp_y[j]+=dataset[i][1]
                    temp_count[j]+=1
        for j in range(k):
            if temp_count[j]==0:
                find=True
                break
            temp_y[j]/=temp_count[j]
            temp_x[j]/=temp_count[j]
            init_ofcenterK[j][0]=temp_x[j]
            init_ofcenterK[j][1]=temp_y[j]
    return clussterAssment,init_ofcenterK,dataset

def k_means_bi(k,dataset):
    init_ofcenterK = get_CenterofK(dataset, k)
    m=shape(dataset)[0]
    clussterAssment = np.zeros((m, 2))
    find =True
    while find:
        find=False
        for i in range(m):
            minIndex=-1
            mindist=inf
            for j in range(k):
                temp_dist=dis_of_2vec(dataset[i],init_ofcenterK[j])
                if temp_dist<mindist:
                    mindist=temp_dist
                    minIndex=j
            if clussterAssment[i][0]!=minIndex: find=True
            clussterAssment[i][0],clussterAssment[i][1]=minIndex,mindist
        temp_x=[0]*k
        temp_y=[0]*k
        temp_count=[0]*k
        for i in range(m):
            for j in range(k):
                if clussterAssment[i][0]==j:
                    temp_x[j]+=dataset[i][0]
                    temp_y[j]+=dataset[i][1]
                    temp_count[j]+=1
        #if find or np.sum(clussterAssment[:,1])<1000000: find=False_
        for j in range(k):
            if temp_count[j]==0:
                break
            temp_y[j]/=temp_count[j]
            temp_x[j]/=temp_count[j]
            init_ofcenterK[j][0]=temp_x[j]
            init_ofcenterK[j][1]=temp_y[j]
    '''
    SSE=0
    SSE_list2=[0]*2
    for i in range(m):
        for j in range(k):
            if clussterAssment[i][0]==j:
                SSE+=dis_of_2vec(dataset[i],init_ofcenterK[j])
                SSE_list2[j]+=dis_of_2vec(dataset[i],init_ofcenterK[j])'''
    SSE=np.sum(clussterAssment[:,1])
    SSE_list2=[0]*2
    for i in range(m):
        if clussterAssment[i][0]==0: SSE_list2[0]+=clussterAssment[i][1]
    SSE_list2[1]=SSE-SSE_list2[0]
    #print len(clussterAssment)
    return clussterAssment,init_ofcenterK,SSE,SSE_list2

def bi_Kmeans(k, index):
    #x1, y1, dataset = creat_dataset()
    x1,y1,dataset=init_dataset(index)
    dataset1=copy.deepcopy(dataset)
    center_of_k_init=np.mean(dataset,axis=0).tolist()     #初始化质心
    list_of_center=[]                                    #存放K个质心
    list_of_center.append(center_of_k_init)
    m=shape(dataset)[0]
    cluster=np.zeros((m,2))     #第一位存储质心，第二位存储该点距质心的距离
    sse=0
    for i in range(m):
        cluster[i][1] = dis_of_2vec(center_of_k_init, dataset[i])
        sse+=cluster[i][1]
    count=1
    test1=0
    while count<k:
        index=-1
        test1+=1
        #print (count,",",test1)
        min_SSE=inf
        for i in range(count):
            dataset_temp=[]
            SSE_left=0
            for j in range(m):
                if cluster[j][0]==i: dataset_temp.append(dataset[j])
                if cluster[j][0]!=i: SSE_left+=cluster[j][1]
            dataset_temp=np.array(dataset_temp)
            if np.shape(dataset_temp)[0]!=0:
                cluster_kmeans,list_of2center,SSE,SSE_list2=k_means_bi(2,dataset_temp)
            #print(count,min_SSE,SSE,SSE_left)
            if SSE+SSE_left<min_SSE:
                min_SSE=SSE+SSE_left
                cluster_kmeans_copy=cluster_kmeans
                list_of2center_copy=list_of2center
                index=i
        if index!=-1:
            tip=0
            for i in range(m):
                if cluster[i][0]==index and tip<len(cluster_kmeans_copy):
                    cluster[i][0]=cluster[i][0] if cluster_kmeans_copy[tip][0]==0 else count
                    cluster[i][1]=cluster_kmeans_copy[tip][1]
                    tip+=1
            count+=1
            list_of_center[index]=list_of2center_copy[0]
            list_of_center.append(list_of2center_copy[1])
            if count==k:
                sse=min_SSE
            #print list_of2center

    return cluster,list_of_center,dataset,sse

def generate_kmeans(k,index):
    #a,b,c = k_means(3)
    #a,b,c,d=bi_Kmeans(k,index)
    #print d
    return bi_Kmeans(k, index)
    '''
    for i in range(3000):
        if a[i][0] == 0:
            plt.plot(c[i][0], c[i][1], 'o', color='b')
        if a[i][0] == 1:
            plt.plot(c[i][0], c[i][1], 'o', color='r')
        if a[i][0] == 2:
            plt.plot(c[i][0], c[i][1], 'o', color='y')
        if a[i][0] == 3:
            plt.plot(c[i][0], c[i][1], 'o', color='c')
        if a[i][0] == 4:
            plt.plot(c[i][0], c[i][1], 'o', color='k')
    b = np.transpose(b)
    plt.plot(b[0], b[1], 'o', color='g')
    plt.title("Bi-Kmeans clustering")
    plt.show()
    '''
if __name__ == '__main__':
    generate_kmeans(5,0)
    #creat_dataset()


# -*- coding: UTF-8 -*-
import numpy as np
import random
import Noise_distribution
import operator

def synthetic_data(d,k):
    p_distri,n,Pi_set,N,A_num_set=Noise_distribution.build_nosis_distr(k,d,0.56)
    lookup_Atr={}
    for i in range(d):
        lookup_Atr[N[i][0]]=i
    dataset_new=[]
    dataset_new1=np.zeros((n,d))
    for count in range(n):
        temp=[]
        num_pi=0
        for i in range(d):
            if i==0 :
                r = np.random.rand()
                sum_p=0
                random_tip=0
                while True:
                    sum_p+=p_distri[i][0][random_tip]
                    if sum_p>=r:break
                    random_tip+=1
                if random_tip<len(p_distri[i][0]):
                    temp.append(random_tip)
                    dataset_new1[count][N[i][0]]=random_tip
                else:
                    temp.append(random_tip-1)
                    dataset_new1[count][N[i][0]]=random_tip-1
                #if count:
                    #print ("i is ",i," temp is ",temp)
                for index,item in enumerate(Pi_set[i+1]):
                    if operator.eq(item,temp):
                        num_pi=index
                        #if count:
                            #print ("i is ",i," num pi is ",num_pi)
                        break
            else:
                r = np.random.rand()
                sum_p = 0
                random_tip = 0
                tip=0
                total_p=0    #对条件分布做归一化处理
                x, y = np.shape(p_distri[i])
                #if count:
                    #print ("the last num pi is ",num_pi,"len p now is ",len(p_distri[i][0]))
                    #print ("x is ",x," y is ",y)
                    #if num_pi>=y:
                        #print ("yes "," num_pi is ",num_pi," y is",y)
                while tip<x:
                    total_p+=p_distri[i][tip][num_pi]
                    tip+=1
                while random_tip<x:
                    temp1=p_distri[i][random_tip][num_pi]/float(total_p)
                    sum_p+=temp1
                    if sum_p>=r:break
                    random_tip+=1
                if random_tip<x:
                    temp.append(random_tip)
                    dataset_new1[count][N[i][0]]=random_tip
                else:
                    temp.append(random_tip-1)
                    dataset_new1[count][N[i][0]]=random_tip-1
                #if count:
                    #print ("i is ",i," temp is ",temp)
                if i<k:
                    for index,item in enumerate(Pi_set[i+1]):
                        if operator.eq(item,temp):
                            #if count:
                                #print ("yes,index is ",index)
                            num_pi=index
                            break
                if i>=k and i<d-1:
                    temp2=[]
                    '''
                    print ("N is",N[i+1][1][0])
                    print (N[i+1][1])
                    print (lookup_Atr[N[i+1][1][0]])
                    print ("temp is ",temp)
                    '''
                    for j in range(k):
                        temp2.append(temp[lookup_Atr[N[i+1][1][j]]])
                    for index,item in enumerate(Pi_set[i+1]):
                        if operator.eq(temp2,item):
                            num_pi=index
                            #if count:
                                #print ("yes,index is ", index,"len pi set is",len(Pi_set[i+1]))
                            break
        dataset_new.append(temp)
    dataset_new=np.array(dataset_new)
    #print dataset_new
    dataset_new1=np.array(dataset_new1,dtype=int)
    dataset_new = np.array(dataset_new, dtype=int)
    np.savetxt("dataset_new.txt", dataset_new1, delimiter=",", fmt="%d")

if __name__ == '__main__':
    synthetic_data(14,5)

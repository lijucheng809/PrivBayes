# -*- coding: UTF-8 -*-
import numpy as np
import random
import itertools
import copy
import operator
def readD2array():
    filename="dataset_binary.txt"
    file=open(filename)
    dataset=[]
    for line in file.readlines():
        line_list=line[:].split(',')
        dataset.append(line_list)
    file.close()
    dataset=np.array(dataset,dtype=int)
    #print dataset
    return dataset

def get_k(d,theta,epislon,n):
    right=n*epislon/theta
    k=1
    while k<d:
        if (d-k)*np.power(2,k+2)>=right:
            break
        else: k+=1
    return k
def get_pi_set(k, pi_init, pi_set, pi_temp):
    if k==len(pi_init):
        temp=copy.deepcopy(pi_temp)
        pi_set.append(temp)
    else:
        for i in range(len(pi_init[k])):
            pi_temp[k]=pi_init[k][i]
            get_pi_set(k + 1, pi_init, pi_set, pi_temp)

def mutual_information(Ax,dataset,pi_set,AP):
    joint_set=[]
    lookup_Ax={}
    lookup_pi={}
    lookup_jonit_x_pi={}
    joint_distributes = np.zeros((len(Ax), len(pi_set)))  # 存放联合概率分布集合
    pi_distributes = np.zeros((1, len(pi_set)))  # 存放父节点的边际概率分布集合
    Ax_distributes=np.zeros((1,len(Ax)))
    n_total=np.shape(dataset)[0]
    for index_Ax,item_Ax in enumerate(Ax):
        lookup_Ax[item_Ax]=index_Ax
        for index_pi_set,item_pi_set in enumerate(pi_set):
            if index_Ax==0:
                temp1=""
                for item_pi in item_pi_set:
                    temp1+=str(item_pi)
                lookup_pi[temp1]=index_pi_set
            temp_joint=copy.deepcopy(item_pi_set)
            temp_joint.insert(0,item_Ax)
            temp2=""
            for item_temp_joint in temp_joint:
                temp2+=str(item_temp_joint)
            lookup_jonit_x_pi[temp2]=index_pi_set
            joint_set.append(temp_joint)
    for item in dataset:
        Ax_distributes[0][lookup_Ax[item[AP[0]]]]+=1
        key_pi=""
        key_Ax_pi=""+str(item[AP[0]])
        for j in AP[1]:
            key_pi+=str(item[j])
            key_Ax_pi+=str(item[j])
        pi_distributes[0][lookup_pi[key_pi]]+=1
        joint_distributes[lookup_Ax[item[AP[0]]]][lookup_jonit_x_pi[key_Ax_pi]]+=1

    joint_distributes=joint_distributes/n_total
    pi_distributes=pi_distributes/n_total
    Ax_distributes=Ax_distributes/n_total

    joint_distributes=np.mat(joint_distributes,dtype=float)
    pi_distributes=np.mat(pi_distributes,dtype=float)
    Ax_distributes=np.mat(Ax_distributes,dtype=float)
    I_Ax_pi_mat=np.log2(np.multiply(joint_distributes,1/(Ax_distributes.T*pi_distributes)))
    I_Ax_pi_mat[np.isnan(I_Ax_pi_mat)]=0
    I_Ax_pi_mat[np.isinf(I_Ax_pi_mat)]=0
    I_Ax_pi_mat[np.isneginf(I_Ax_pi_mat)]=0
    I_Ax_pi_mat=np.multiply(I_Ax_pi_mat,joint_distributes)
    I_Ax_pi_total=I_Ax_pi_mat.sum()
    return I_Ax_pi_total,joint_distributes,pi_distributes


def exponent_mechanism(mute_info_list,n_dataset,epsilon,d):
    sensitivity=2./n_dataset*np.log2(float((n_dataset+1))/2)+float(n_dataset-1)/n_dataset*np.log2(float(n_dataset+1)/(n_dataset-1))
    epsilon_single=epsilon/(d-1)
    sum=0
    sum_exp_mecha=0
    exponent_mechanism_list=[]
    for I_Ax_pi in mute_info_list:
        temp=np.exp(0.5*I_Ax_pi*epsilon_single/sensitivity)
        exponent_mechanism_list.append(temp)
        sum+=temp
    exponent_mechanism_list=np.array(exponent_mechanism_list)
    exponent_mechanism_list=exponent_mechanism_list/sum
    r=np.random.rand()
    num_list=0
    while True:
        sum_exp_mecha+=exponent_mechanism_list[num_list]
        if sum_exp_mecha>r:break
        num_list+=1
    return num_list

def generate_N(d,theta,epsilon): #d:属性个数
    dataset=readD2array()
    n_dataset=np.shape(dataset)[0]
    k = get_k(d,theta, epsilon,n_dataset)
    print k
    #print k
    A_set=[]
    Ax_Pi_distrs=[]
    Pi_distrs=[]
    Pi_distrs.append([])
    Pi_set=[]
    Pi_set.append([])
    for i in range(d):
       A_set.append(i)
    #A1=random.randint(0,d-1)
    A1=0
    #print A1
    A_set.pop(A1)
    V=[]
    V.append(A1)
    N=[]
    N.append([A1])
    filename = 'adult_property.data'
    file = open(filename)
    dataset_Atr = []
    for line in file.readlines():
        line = line.strip('\n')
        line_list = line[:].split(',')
        dataset_Atr.append(line_list)
    dataset_Atr.append(['<=50K', '>50K'])
    A_num_set=[]
    lookup_Acontinues={0:5,2:7,4:5,10:6,11:2,12:7}
    for i in range(len(dataset_Atr)):
        if i != 0 and i != 2 and i != 10 and i != 11 and i != 12 and i != 4:
            A_num_set.append(len(dataset_Atr[i]))
        else: A_num_set.append(lookup_Acontinues[i])

    '''生成A1的联合（边际）分布'''
    joint_A1=np.zeros((1,A_num_set[A1]))
    for item in dataset:
        joint_A1[0][item[A1]]+=1
    joint_A1=joint_A1/n_dataset
    Ax_Pi_distrs.append(joint_A1)

    for i in range(len(A_set)):
        AP_list=[]
        mute_info_list=[]
        joint_distr_list=[]
        pi_distr_list=[]
        pi_set_list=[]
        #I_max=0
        #I_max_pair=[]
        for Ai in A_set:
            if len(V)<=k:
                AP_list.append([Ai,V])
            else:
                for pai in itertools.combinations(V,k):
                    pai=list(pai)
                    AP_list.append([Ai,pai])
        if i==0:print AP_list
        #print ("len AP_list is ",len(AP_list))
        for AP in AP_list:
            pi_set=[]            #存放所有可能的父节点取值，通过递归实现
            pi=AP[1]             #父节点属性集合
            pi_init=[]           #保存父节点中各个属性的可能取值
            pi_temp=[0]*len(pi)  #存放递归过程中的每个可能的父节点取值
            for i in range(len(pi)):
                temp=[]
                for j in range(int(A_num_set[pi[i]])):
                    temp.append(j)
                pi_init.append(temp)
            get_pi_set(0,pi_init,pi_set,pi_temp)
            pi_set_list.append(pi_set)
            #print ("len pi set is ",len(pi_set))
            Ax=[]               #保存当前结点属性取值的集合
            for i in range(int(A_num_set[AP[0]])):
                Ax.append(i)
            mute_info, joint_distributes, pi_distributes = mutual_information(Ax, dataset, pi_set, AP)
            mute_info_list.append(mute_info)
            joint_distr_list.append(joint_distributes)
            pi_distr_list.append(pi_distributes)
            '''不含噪声的AP对
            if I_max<mute_info:
                I_max=mute_info
                I_max_pair=AP
            '''
        '''含有噪声的AP对'''
        index_mute_info_list=exponent_mechanism(mute_info_list,n_dataset,epsilon,d)
        I_max_pair=AP_list[index_mute_info_list]
        Ax_Pi_distrs.append(joint_distr_list[index_mute_info_list])
        Pi_distrs.append(pi_distr_list[index_mute_info_list])
        Pi_set.append(pi_set_list[index_mute_info_list])
        I_max_pair_copy=copy.deepcopy(I_max_pair)
        N.append(I_max_pair_copy)
        V.append(I_max_pair_copy[0])
        A_set.remove(I_max_pair_copy[0])
    return dataset,Ax_Pi_distrs,Pi_distrs,Pi_set,N,A_num_set

if __name__ == '__main__':
    generate_N(14,7,0.12)

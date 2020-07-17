# -*- coding: UTF-8 -*-
import numpy as np
import copy
def readdata(filename):
    file=open(filename)
    dataset=[]
    for line in file.readlines():
        line=line.strip('\n')
        list_line=line[:].split(',')
        dataset.append(list_line)
    dataset=np.array(dataset)
    return dataset
def jonint_distribution(joint_set,dataset_a,a):
    lookup_joint={}
    joint_distri=np.zeros((1,len(joint_set)))
    for index_joint_set, item_joint_set in enumerate(joint_set):
        temp1 = ""
        for item_pi in item_joint_set:
            temp1 += str(item_pi)
        lookup_joint[temp1] =  index_joint_set
    for item in dataset_a:
        key_item=""
        for i in range(a):
            key_item+=str(item[i])
        joint_distri[0][lookup_joint[key_item]]+=1
    joint_distri=joint_distri/len(dataset_a)
    return joint_distri

def joint_Set(k,joint_init,joint_set,joint_temp):
    if k==len(joint_init):
        temp=copy.deepcopy(joint_temp)
        joint_set.append(temp)
    else:
        for i in range(len(joint_init[k])):
            joint_temp[k]=joint_init[k][i]
            joint_Set(k + 1, joint_init, joint_set, joint_temp)
def A_Num_Set(filename):
    file = open(filename)
    dataset_Atr = []
    for line in file.readlines():
        line = line.strip('\n')
        line_list = line[:].split(',')
        dataset_Atr.append(line_list)
    dataset_Atr.append(['<=50K', '>50K'])
    A_num_set = []
    lookup_Acontinues={0:5,2:7,4:5,10:6,11:2,12:7}
    for i in range(len(dataset_Atr)):
        if i != 0 and i != 2 and i != 10 and i != 11 and i != 12 and i != 4:
            A_num_set.append(len(dataset_Atr[i]))
        else:
            A_num_set.append(lookup_Acontinues[i])
    return A_num_set

def joint_distribution_old_or_new_2(a, d, dataset_old_ornew):
    A_num_set=A_Num_Set("adult_property.data")
    joint_distri_set=[]
    for i in range(d-1):
        for j in range(i+1,d):
            dataset_a=[]
            x= dataset_old_ornew[:, i]
            y= dataset_old_ornew[:, j]
            dataset_a.append(x)
            dataset_a.append(y)
            dataset_a=np.transpose(np.array(dataset_a))
            joint_init=[]    #存放联合分布中各个属性可能的取值
            joint_temp=[0]*a  #存放递归中每层联合分布的取值
            joint_set=[]
            for index_a in range(a):
                temp=[]
                if index_a==0:
                    for index_0 in range(int(A_num_set[i])):
                        temp.append(index_0)
                    joint_init.append(temp)
                if index_a==1:
                    for index_0 in range(int(A_num_set[j])):
                        temp.append(index_0)
                    joint_init.append(temp)
            joint_Set(0,joint_init,joint_set,joint_temp)
            joint_distri_set.append(jonint_distribution(joint_set,dataset_a,a))
    return joint_distri_set

def joint_distribution_old_or_new_3(a, d, dataset_old_ornew):
    A_num_set = A_Num_Set("adult_property.data")
    joint_distri_set = []
    for i in range(d - 2):
        for j in range(i + 1, d-1):
            for k in range(j+1,d):
                dataset_a = []
                x = dataset_old_ornew[:, i]
                y = dataset_old_ornew[:, j]
                z = dataset_old_ornew[:,k]
                dataset_a.append(x)
                dataset_a.append(y)
                dataset_a.append(z)
                dataset_a = np.transpose(np.array(dataset_a))
                joint_init = []  # 存放联合分布中各个属性可能的取值
                joint_temp = [0] * a  # 存放递归中每层联合分布的取值
                joint_set = []
                for index_a in range(a):
                    temp = []
                    if index_a == 0:
                        for index_0 in range(int(A_num_set[i])):
                            temp.append(index_0)
                        joint_init.append(temp)
                    if index_a == 1:
                        for index_0 in range(int(A_num_set[j])):
                            temp.append(index_0)
                        joint_init.append(temp)
                    if index_a==2:
                        for index_0 in range(int(A_num_set[k])):
                            temp.append(index_0)
                        joint_init.append(temp)
                joint_Set(0, joint_init, joint_set, joint_temp)
                joint_distri_set.append(jonint_distribution(joint_set, dataset_a, a))
    return joint_distri_set

def main(a,d):
    dataset_old = readdata("dataset_binary.txt")
    dataset_new = readdata("dataset_new.txt")
    joint_distri_set_old=joint_distribution_old_or_new_3(a,d,dataset_old)
    joint_distri_set_new = joint_distribution_old_or_new_3(a, d, dataset_new)
    n=len(joint_distri_set_new)
    aver_var_dist=0
    aver_distance=0
    for i in range(n):
        joint_distri_set_old[i]=np.fabs(joint_distri_set_old[i]-joint_distri_set_new[i])
        aver_var_dist+=np.max(joint_distri_set_old[i])
    aver_var_dist/=n
    print aver_var_dist
main(2,14)

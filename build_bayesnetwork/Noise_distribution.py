# -*- coding: UTF-8 -*-
import numpy as np
import build_N
def build_nosis_distr(k,d,epsilon):
    dataset, Ax_Pi_distrs, Pi_distrs,Pi_set,N,A_num_set=build_N.generate_N(d,7,0.24)
    '''
    for i in range(14):
        Ax_Pi_distri = np.array(Ax_Pi_distrs[i])
        Pi_distri = np.array(Pi_distrs[i])
        x1,y1=np.shape(Ax_Pi_distri)
        x2=len(Pi_distri)
        print ("x1 is ",x1," y1 is ",y1)
        print ("x2 is ",x2)
    '''
    n=np.shape(dataset)[0]
    Lamba = 2 * d / (n * epsilon)
    p_distri = []
    for index in range(d):
        sum_p=0
        Ax_Pi_distri = np.array(Ax_Pi_distrs[index])
        #Pi_distri = np.array(Pi_distrs[index])
        for i in range(np.shape(Ax_Pi_distri)[0]):
            for j in range(np.shape(Ax_Pi_distri)[1]):
                temp = Ax_Pi_distri[i][j] + np.random.laplace(0, Lamba)
                if temp < 0:
                    Ax_Pi_distri[i][j] = 0
                else:
                    Ax_Pi_distri[i][j] = temp
                sum_p += Ax_Pi_distri[i][j]
        Ax_Pi_distri = Ax_Pi_distri / sum_p
        p_distri.append(Ax_Pi_distri)

    '''
    n=np.shape(dataset)[0]
    Lamba=2*(d-k)/(n*epsilon)
    p_distri=[]   #存放低维概率分布
    for index in range(k,d):
        sum_p=0   #概率分布归一化处理
        Ax_Pi_distri=np.array(Ax_Pi_distrs[index])
        Pi_distri=np.array(Pi_distrs[index])
        for i in range(np.shape(Ax_Pi_distri)[0]):
            for j in range(np.shape(Ax_Pi_distri)[1]):
                temp=Ax_Pi_distri[i][j]+np.random.laplace(0,Lamba)
                if temp<0: Ax_Pi_distri[i][j]=0
                else:Ax_Pi_distri[i][j]=temp
                sum_p+=Ax_Pi_distri[i][j]
        Ax_Pi_distri=Ax_Pi_distri/sum_p
        for i in range(np.shape(Ax_Pi_distri)[0]):
            for j in range(np.shape(Ax_Pi_distri)[1]):
                if Pi_distri[0][j]!=0:
                    Ax_Pi_distri[i][j]=Ax_Pi_distri[i][j]/Pi_distri[0][j]
        p_distri.append(Ax_Pi_distri)

    for index in range(k-1,-1,-1):
        Ax_Pi_distri=np.array(Ax_Pi_distrs[index])
        Pi_distri = np.array(Pi_distrs[index])
        temp=np.sum(Ax_Pi_distrs[index+1],axis=0)
        x,y=np.shape(Ax_Pi_distrs[index])
        temp=np.array(temp.reshape(x,y))
        if index==0: Pi_distrs.insert(0,temp)
        else:
            for i in range(np.shape(Ax_Pi_distri)[0]):
                for j in range(np.shape(Ax_Pi_distri)[1]):
                    if Pi_distri[0][j]!=0:
                        temp[i][j]=temp[i][j]/Pi_distri[0][j]
        p_distri.insert(0,temp)
    print p_distri
    '''
    '''
    Q=[]
    for i in range(k):
        Q.append(Ax_Pi_distrs[k])
    Q.extend(p_distri)
    p_distri=Q
    '''
    p_distri=np.array(p_distri)
    Pi_set=np.array(Pi_set)
    #print Pi_set
    return p_distri,n,Pi_set,N,A_num_set

if __name__ == '__main__':
    build_nosis_distr(3,14,0.4)
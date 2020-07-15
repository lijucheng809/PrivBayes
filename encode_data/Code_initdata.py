# -*- coding: UTF-8 -*-
import numpy as np
import init_adultdata
import bi_kmeans
def discre_continuous(num,index):          #num为区间数量
    cluster,list_of_center,dataset,sse=bi_kmeans.generate_kmeans(num,index)
    data=[]
    m=np.shape(cluster)[0]
    for i in range(num):
        temp=[]
        for j in range(m):
            if cluster[j][0]==i:
                temp.append(dataset[j][0])
        data.append(temp)
    data=np.array(data)
    data_min_max=[]
    for i in range(num):
        temp=[]
        temp.append(np.min(data[i]))
        temp.append(np.max(data[i]))
        data_min_max.append(temp)
    data_min_max=np.array(data_min_max,dtype=int)

    '''对分好组的进行从小到大排序'''
    data_min_max=np.transpose(data_min_max)
    count=sorted(range(len(data_min_max[0])),key=lambda k:data_min_max[0][k])
    data_min_max_new=[]
    data_min_max = np.transpose(data_min_max)
    for i in range(num):
        data_min_max_new.append(data_min_max[count[i]])
    data_min_max_new=np.array(data_min_max_new,dtype=int)
    print data_min_max_new
    #print sse
    return data_min_max_new

def encode(num,num_group_disc):
    '''
    :num: 属性个数
    :num_group_disc: 将离散数据进行进组后的数据
    0  2 10 11 12 为continuous
    age: continuous.
    work-class: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    '''
    filename='adult_property.data'
    file=open(filename)
    dataset_Atr=[]
    for line in file.readlines():
        line=line.strip('\n')
        line_list=line[:].split(',')
        dataset_Atr.append(line_list)
    dataset_Atr.append(['<=50K','>50K'])
    '''建立属性字典集'''
    lookup_arrtribut={}
    for i in range(len(dataset_Atr)):
        if i != 0 and i != 2 and i != 10 and i != 11 and i != 12 and i!=4:
            for j,item in enumerate(dataset_Atr[i]):
                lookup_arrtribut[item]=j

    '''建立离散属性集合'''
    data_discre=[]
    for i in range(14):
        if i==0:
            data_discre.append(discre_continuous(5,i))
            continue
        if i==2:
            data_discre.append(discre_continuous(7,i))
            continue
        if i==4:
            data_discre.append(discre_continuous(5,i))
            continue
        if i==10:
            data_discre.append(discre_continuous(6,i))
            continue
        if i==11:
            data_discre.append(discre_continuous(2,i))
            continue
        if i==12:
            data_discre.append(discre_continuous(7,i))
            continue
        '''
        if i==0 or i==2 or i==4 or i==10 or i==11 or i==12 :
            data_discre.append(discre_continuous(num_group_disc,i))
            continue
        '''
        data_discre.append(dataset_Atr[i])
    '''对原始属性集进行编码'''
    dataset=init_adultdata.init_AdultData()
    m=np.shape(dataset)[0]
    dataset_01=[]
    for i in range(m):
        temp=[]
        for j in range(num):
            if j ==0 or j ==2 or j == 4 or j == 10 or j == 11 or j == 12 :
                tip=0
                count=0
                data=int(dataset[i][j])
                while count<len(data_discre):
                    if data>=data_discre[j][tip][0] and data<=data_discre[j][tip][1]:
                        temp.append(tip)
                        break
                    else:tip+=1
                    count+=1
            else:
                temp.append(lookup_arrtribut[dataset[i][j]])
        if i==0: print temp
        dataset_01.append(temp)
    dataset_01=np.array(dataset_01,dtype=int)
    np.savetxt("dataset_binary.txt",dataset_01,delimiter=",",fmt="%d")
    print dataset_01[1]

if __name__ == '__main__':
    encode(14,8)
    '''
    for i in range(1,12):
        discre_continuous(i,12)
    '''
'''生成laplace随机数
a=np.random.laplace(0,0.1)
'''
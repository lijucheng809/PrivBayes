# -*- coding: UTF-8 -*-
import numpy as np

def get_adult():
    filename = 'adult.data'
    file=open(filename)
    dataset=[]
    for line in file.readlines():
        line = line.strip('\n')
        line_list=line[:].split(',')
        dataset.append(line_list)
    dataset=np.array(dataset)
    dataset=np.transpose(dataset)
    file.close()
    return dataset

def init_AdultData():
    dataset = np.transpose(get_adult()).tolist()
    m = np.shape(dataset)[0]
    dataset_new = []
    for i in range(m):
        if dataset[i][1] != '?' and dataset[i][6] != '?' and dataset[i][13] != '?' and dataset[i][13]=='United-States':
            dataset_new.append(dataset[i])  #去除问题数据，国籍只保留United-States
    dataset_new = np.array(dataset_new)
    dataset_new=np.delete(dataset_new,13,axis=1)  #删除国籍属性
    '''测试集合使用
    dataset_new=np.transpose(dataset_new)
    a=set(dataset_new[3])
    print a
    '''
    return  dataset_new

if __name__ == '__main__':
    init_AdultData()

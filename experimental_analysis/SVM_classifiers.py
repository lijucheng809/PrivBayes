# -*- coding: UTF-8 -*-
import sklearn
from sklearn import svm
import numpy as np
def readdata():
    filename="dataset_new.txt"
    file=open(filename)
    dataset=[]
    for line in file.readlines():
        line=line.strip('\n')
        list_line=line[:].split(',')
        dataset.append(list_line)
    dataset=np.array(dataset)
    y=dataset[:,3]

    for i in range(len(y)):
        if y[i]=='0' or y[i]=='10' or y[i]=='13':
            y[i]=0
        else:y[i]=1

    '''
    for i in range(len(y)):
        if y[i]=='1' or y[i]=='2' or y[i]=='3' or y[i]=='5':
            y[i]=0
        else:y[i]=1
    '''
    x=np.delete(dataset,3,axis=1)
    return x,y
def SVM():
    x,y=readdata()
    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x, y, random_state=1,
                                                                                              train_size=0.8,
                                                                                              test_size=0.2)
    classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
    classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
    print classifier.score(test_data, test_label)

if __name__ == '__main__':
    SVM()

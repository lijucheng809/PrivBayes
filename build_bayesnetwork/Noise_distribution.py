# -*- coding: UTF-8 -*-
import numpy as np
import build_N
def build_nosis_distr(k,d,epsilon):
    dataset, Ax_Pi_distrs, Pi_distrs,Pi_set,N,A_num_set=build_N.generate_N(d,7,0.24)
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

    p_distri=np.array(p_distri)
    Pi_set=np.array(Pi_set)
    #print Pi_set
    return p_distri,n,Pi_set,N,A_num_set

if __name__ == '__main__':
    build_nosis_distr(3,14,0.4)

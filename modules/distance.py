import numpy as np

def dist(p1, p2):
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]
    #if x2 < x1:
    #    return 10000000
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def nearest_pt(boundary, mu, idxi, idxj):
    pt = [boundary[idxi, idxj], mu[idxi]]
    if idxi >= 1:
        ilower = idxi - 1
    else:
        ilower = 0
    if idxj >= 1:
        jlower = idxj-1
    else:
        jlower = 0
    if idxi < boundary.shape[0]-15:
        ihigher = idxi+15
    else:
        ihigher = boundary.shape[0]-1
    if idxj < boundary.shape[1]-1:
        jhigher = idxj+1
    else:
        jhigher = boundary.shape[1]-1

    #print(ihigher, ilower)
    #print(jhigher, jlower)
    distance = np.zeros(((ihigher-ilower), (jhigher-jlower)))
    #print(distance.shape)
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            #print(i,j)
            D = dist([boundary[ilower+i,jlower+j], mu[ilower+i]], pt)
            if D == 0:
                distance[i, j] = 100000000
            else:
                distance[i,j] = D

    mini = 0
    minj = 0
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            if distance[i,j] < distance[mini, minj]:
                mini = i
                minj = j
    #print(mini, minj)
    NP = boundary[ilower+mini,jlower+minj]
    return NP, ilower+mini,jlower+minj

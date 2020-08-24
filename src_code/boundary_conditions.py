# -*-coding:Latin-1 -*

import numpy as np
from numerical_integration import *

def SMBC(F,X,T,ie,b,j,fixedx):
    b=b.astype(int)
    j=j-1
    points, weights = GAUSS(6)
    FEBC = np.zeros((2,1))
    x = X[b[[ie,ie+1]]-1,j]
    x2 = x[1]
    x1 = x[0]
    xm = 0.5 * (x2 + x1)

    bingo = 0
    bingo1 = 0

    for i in np.arange(0,T.shape[0]):
        if bingo !=0:
            break
        for h in [0,1,2]:
            if np.array_equal(X[b[ie]-1,:], X[T[i,h],:]):
                if np.isin(x1,X[T[i,:],j]):
                    if np.isin(x2, X[T[i, :], j]):
                        bingo=i
                        bingo1=h
                        break

    bingo2 = 0
    bingo21 = 0

    for i in np.arange(0, T.shape[0]):
        if bingo2 !=0:
            break
        for h in [0, 1, 2]:
            if np.array_equal(X[b[ie+1]-1, :], X[T[i, h], :]):
                if np.isin(x1, X[T[i, :], j]):
                    if np.isin(x2, X[T[i, :], j]):
                        bingo2 = i
                        bingo21 = h
                        break

    xe = X[T[bingo, :], :]

    b = np.zeros((1, 3))
    b[0,0] = xe[1, 1] - xe[2, 1]
    b[0,1] = xe[2, 1] - xe[0, 1]
    b[0,2] = xe[0, 1] - xe[1, 1]

    c = np.zeros((1, 3))
    c[0,0] = xe[2, 0] - xe[1, 0]
    c[0,1] = xe[0, 0] - xe[2, 0]
    c[0,2] = xe[1, 0] - xe[0, 0]
    delta = 0.5 * (b[0,0] * c[0,1] - b[0,1] * c[0,0])

    a = np.zeros((1, 3))
    a[0,0] = xe[1, 0]*xe[2, 1] - xe[2, 0]*xe[1, 1]
    a[0,1] = xe[2, 0]*xe[0, 1] - xe[0, 0]*xe[2, 1]
    a[0,2] = xe[0, 0]*xe[1, 1] - xe[1, 0]*xe[0, 1]

    if j==0:
        N= lambda x : (a[0,bingo1]+b[0,bingo1]*x+c[0,bingo1]*fixedx)/(2*delta)
    else:
        N= lambda y:  (a[0,bingo1]+b[0,bingo1]*fixedx+c[0,bingo1]*y)/(2*delta)

    ii=0
    while ii< points.shape[0]:
     FEBC[0] += weights[ii] * N((0.5 * (x2 - x1)) * points[ii] + xm) * F(((x2 - x1) / 2) * (points[ii]) + xm) * (0.5 * (x2 - x1))
     ii+=1

    xe = X[T[bingo2, :], :]

    b = np.zeros((1, 3))
    b[0,0] = xe[1, 1] - xe[2, 1]
    b[0,1] = xe[2, 1] - xe[0, 1]
    b[0,2] = xe[0, 1] - xe[1, 1]

    c = np.zeros((1, 3))
    c[0,0] = xe[2, 0] - xe[1, 0]
    c[0,1] = xe[0, 0] - xe[2, 0]
    c[0,2] = xe[1, 0] - xe[0, 0]
    delta = 0.5 * (b[0,0] * c[0,1] - b[0,1] * c[0,0])

    a = np.zeros((1, 3))
    a[0,0] = xe[1, 0] * xe[2, 1] - xe[2, 0] * xe[1, 1]
    a[0,1] = xe[2, 0] * xe[0, 1] - xe[0, 0] * xe[2, 1]
    a[0,2] = xe[0, 0] * xe[1, 1] - xe[1, 0] * xe[0, 1]

    if j == 0:
        N = lambda x: (a[0,bingo21] + b[0,bingo21] * x + c[0,bingo21] * fixedx) / (2 * delta)
    else:
        N = lambda y: (a[0,bingo21] + b[0,bingo21] * fixedx + c[0,bingo21] * y) / (2 * delta)

    ii = 0
    while ii < points.shape[0]:
        FEBC[1] += weights[ii] * N((0.5 * (x2 - x1)) * points[ii] + xm) * F(((x2 - x1) / 2) * (points[ii]) + xm) * (0.5 * (x2 - x1))
        ii += 1

    return FEBC


def SMBCM(F,X,T,ie,b,j,fixedx,xy):
    j=j-1
    points, weights = GAUSS(6)
    FEBCM = np.zeros((2,1))
    x = X[b[[ie,ie+1]]-1,j]
    x2 = x[1]
    x1 = x[0]
    xm = 0.5 * (x2 + x1)

    bingo = 0
    bingo1 = 0

    for i in np.arange(0, T.shape[0]):
        if bingo !=0:
            break
        for h in [0, 1, 2]:
            if np.array_equal(X[b[ie ]-1, :] , X[T[i, h], :]):
                if np.isin(x1, X[T[i, :], j]):
                    if np.isin(x2, X[T[i, :], j]):
                        bingo = i
                        bingo1 = h
                        break

    bingo2 = 0
    bingo21 = 0

    for i in np.arange(0, T.shape[0]):
        if bingo2 !=0:
            break
        for h in [0, 1, 2]:
            if np.array_equal(X[b[ie+1]-1, :] , X[T[i, h], :]):
                if np.isin(x1, X[T[i, :], j]):
                    if np.isin(x2, X[T[i, :], j]):
                        bingo2 = i
                        bingo21 = h
                        break

    bingototal = np.array([[bingo, bingo1], [bingo2, bingo21]])

    kk=0
    while kk<2:
     xe=X[T[bingototal[kk,0],:],:]
     b = np.zeros((3, 1))
     b[0,0] = xe[1, 1] - xe[2, 1]
     b[1,0] = xe[2, 1] - xe[0, 1]
     b[2,0] = xe[0, 1] - xe[1, 1]

     c = np.zeros((1, 3))
     c[0,0] = xe[2, 0] - xe[1, 0]
     c[0,1] = xe[0, 0] - xe[2, 0]
     c[0,2] = xe[1, 0] - xe[0, 0]

     cb=np.concatenate((c.T,b),axis=1)

     v12 = xe[1, :] - xe[0, :]
     v13 = xe[2, :] - xe[0, :]
     v23 = xe[2, :] - xe[1, :]

     l = np.zeros((1, 3))
     l[0,0] = np.sqrt(np.dot(v23 , np.array([v23]).T))
     l[0,1] = np.sqrt(np.dot(v13 , np.array([v13]).T))
     l[0,2] = np.sqrt(np.dot(v12 , np.array([v12]).T))

     mu = np.zeros((1, 3))
     mu[0,0] = (m.pow(l[0,2], 2) - m.pow(l[0,1], 2)) / (m.pow(l[0,0], 2))
     mu[0,1] = (m.pow(l[0,0], 2) - m.pow(l[0,2], 2)) / (m.pow(l[0,1], 2))
     mu[0,2] = (m.pow(l[0,1], 2) - m.pow(l[0,0], 2)) / (m.pow(l[0,2], 2))

     a = np.zeros((1, 3))
     a[0,0] = xe[1, 0] * xe[2, 1] - xe[2, 0] * xe[1, 1]
     a[0,1] = xe[2, 0] * xe[0, 1] - xe[0, 0] * xe[2, 1]
     a[0,2] = xe[0, 0] * xe[1, 1] - xe[1, 0] * xe[0, 1]

     delta = 0.5 * (b[0,0] * c[0,1] - b[1,0] * c[0,0])

     A = (1 / (2 * delta)) * np.array([[b[0,0], b[1,0], b[2,0]], [c[0,0], c[0,1], c[0,2]]])

     if j == 0:
         L1 = lambda x: (a[0, 0] + b[0, 0] * x + c[0, 0] * fixedx) / (2 * delta)

         L2 = lambda x: (a[0, 1] + b[1, 0] * x + c[0, 1] * fixedx) / (2 * delta)

         L3 = lambda x: (a[0, 2] + b[2, 0] * x + c[0, 2] * fixedx) / (2 * delta)

     else:
         L1 = lambda y: (a[0, 0] + b[0, 0] * fixedx + c[0, 0] * y) / (2 * delta)

         L2 = lambda y: (a[0, 1] + b[1, 0] * fixedx + c[0, 1] * y) / (2 * delta)

         L3 = lambda y: (a[0, 2] + b[2, 0] * fixedx + c[0, 2] * y) / (2 * delta)

     ii=0

     while ii<len(points):
      xtransform = (0.5 * (x2 - x1)) * points[ii] + xm


      P=np.zeros((9,3))

      P[:,0]=np.array([[1],
                       [0],
                       [0],
                       [L2(xtransform)],
                       [0],
                       [L3(xtransform)],
                       [2 * L2(xtransform) * L1(xtransform) + L1(xtransform) * L2(xtransform) * L3(xtransform) * 3 * (1 - mu[0,2]) - 0.5 * L2(xtransform) ** 2 * L3(xtransform) * (1 + 3 * mu[0,2]) + 0.5 * L3(xtransform) ** 2 * L2(xtransform) * (1 + 3 * mu[0,2])],
                       [L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,0]) + 0.5 * L2(xtransform) ** 2 * L3(xtransform) * 3 * (1 - mu[0,0]) - 0.5 * L3(xtransform) ** 2 * L2(xtransform) * (1 + 3 * mu[0,0])],
                       [L3(xtransform) ** 2 - L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,1]) + 0.5 * L2(xtransform) ** 2 * L3(xtransform) * (1 + 3 * mu[0,1]) + 0.5 * L3(xtransform) ** 2 * L2(xtransform) * 3 * (1 - mu[0,1])]]).T

      P[:,1]=np.array([[0],
                       [1],
                       [0],
                       [L1(xtransform)],
                       [L3(xtransform)],
                       [0],
                       [L1(xtransform) ** 2 - L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,2]) + 0.5 * L3(xtransform) ** 2 * L1(xtransform) * (1 + 3 * mu[0,2]) + 0.5 * L1(xtransform) ** 2 * L3(xtransform) * 3 * (1 - mu[0,2])],
                       [2 * L2(xtransform) * L3(xtransform) + L1(xtransform) * L2(xtransform) * L3(xtransform) * 3 * (1 - mu[0,0]) - 0.5 * L3(xtransform) ** 2 * L1(xtransform) * (1 + 3 * mu[0,0]) + 0.5 * L1(xtransform) ** 2 * L3(xtransform) * (1 + 3 * mu[0,0])],
                       [L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,1]) + 0.5 * L3(xtransform) ** 2 * L1(xtransform) * 3 * (1 - mu[0,1]) - 0.5 * L1(xtransform) ** 2 * L3(xtransform) * (1 + 3 * mu[0,1])]]).T



      P[:,2]=np.array([[0],
                       [0],
                       [1],
                       [0],
                       [L2(xtransform)],
                       [L1(xtransform)],
                       [L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,2]) - 0.5 * L2(xtransform) ** 2 * L1(xtransform) * (1 + 3 * mu[0,2]) + 0.5 * L1(xtransform) ** 2 * L2(xtransform) * 3 * (1 - mu[0,2])],
                       [L2(xtransform) ** 2 - L1(xtransform) * L2(xtransform) * L3(xtransform) * (1 + 3 * mu[0,0]) + 0.5 * L2(xtransform) ** 2 * L1(xtransform) * 3 * (1 - mu[0,0]) + 0.5 * L1(xtransform) ** 2 * L2(xtransform) * (1 + 3 * mu[0,0])],
                       [2 * L3(xtransform) * L1(xtransform) + L1(xtransform) * L2(xtransform) * L3(xtransform) * 3 * (1 - mu[0,1]) - 0.5 * L1(xtransform) ** 2 * L2(xtransform) * (1 + 3*mu[0,1]) + 0.5 * L2(xtransform) ** 2 * L1(xtransform) * (1 + 3 * mu[0,1])]]).T

      if j==1:
       NL = -np.array([[(-c[0,1]*(P[8,0]-P[5,0])-c[0,2]*P[6,0]), (-c[0,1]*(P[8,1]-P[5,1])-c[0,2]*P[6,1]), (-c[0,1]*(P[8,2]-P[5,2])-c[0,2]*P[6,2])],
                       [(-c[0,2]*(P[6,0]-P[3,0])-c[0,0]*P[7,0]), (-c[0,2]*(P[6,1]-P[3,1])-c[0,0]*P[7,1]), (-c[0,2]*(P[6,2]-P[3,2])-c[0,0]*P[7,2])],
                       [(-c[0,0]*(P[7,0]-P[4,0])-c[0,1]*P[8,0]), (-c[0,0]*(P[7,1]-P[4,1])-c[0,1]*P[8,1]), (-c[0,0]*(P[7,2]-P[4,2])-c[0,1]*P[8,2])]])

      else:

       NL = np.array([[(-b[1,0] * (P[8, 0] - P[5, 0]) - b[2,0] * P[6, 0]), (-b[1,0] * (P[8, 1] - P[5, 1]) - b[2,0] * P[6, 1]), (-b[1,0] * (P[8, 2] - P[5, 2]) - b[2,0] * P[6, 2])],
                      [(-b[2,0] * (P[6, 0] - P[3, 0]) - b[0,0] * P[7, 0]), (-b[2,0] * (P[6, 1] - P[3, 1]) - b[0,0] * P[7, 1]), (-b[2,0] * (P[6, 2] - P[3, 2]) - b[0,0] * P[7, 2])],
                      [(-b[0,0] * (P[7, 0] - P[4, 0]) - b[1,0] * P[8, 0]), (-b[0,0] * (P[7, 1] - P[4, 1]) - b[1,0] * P[8, 1]), (-b[0,0] * (P[7, 2] - P[4, 2]) - b[1,0] * P[8, 2])]])


      N = (1 / (2 * delta)) * np.dot(NL,cb[:,j])

      FEBCM[kk,0]+= weights[ii]*N[bingototal[kk,1]]*F(((x2-x1)/2)*(points[ii])+xm)*(0.5*(x2-x1))
      ii+=1
     kk +=1
    return FEBCM

# -*-coding:Latin-1 -*

import numpy as np
import math as m


def strain_calc(X,T,u,v):
    TT=T.shape[0]
    STRAIN=np.zeros((3*TT,1))

    ii=0
    while ii<TT:
        xe = X[T[ii, :], :]
        b = np.zeros((1, 3))
        b[0,0] = xe[1, 1] - xe[2, 1]
        b[0,1] = xe[2, 1] - xe[0, 1]
        b[0,2] = xe[0, 1] - xe[1, 1]

        c = np.zeros((1, 3))
        c[0,0] = xe[2, 0] - xe[1, 0]
        c[0,1] = xe[0, 0] - xe[2, 0]
        c[0,2] = xe[1, 0] - xe[0, 0]

        delta = 0.5 * (b[0,0] * c[0,1] - b[0,1] * c[0,0])
        dN1dx = b[0,0] / (2 * delta)
        dN1dy = c[0,0] / (2 * delta)
        dN2dx = b[0,1] / (2 * delta)
        dN2dy = c[0,1] / (2 * delta)
        dN3dx = b[0,2] / (2 * delta)
        dN3dy = c[0,2] / (2 * delta)

        STRAIN[[ii,ii + TT,ii + 2 * TT]] = np.array([[dN1dx * u[T[ii,0]] + dN2dx * u[T[ii,1]] + dN3dx * u[T[ii, 2]]],
                                                     [(dN1dy * v[T[ii, 0]] + dN2dy * v[T[ii, 1]] + dN3dy * v[T[ii,2]])],
                                                     [(dN1dy * u[T[ii,0]] + dN2dy * u[T[ii, 1]] + dN3dy * u[T[ii, 2]] + dN1dx * v[T[ii, 0]] + dN2dx * v[T[ii, 1]] + dN3dx * v[T[ii, 2]])]])
        ii+=1
    return STRAIN

def k_calc(X,T,w,thetax,thetay,ie):

    k=np.zeros((9,1))
    xe = X[T[ie, :], :]
    b = np.zeros((1, 3))
    b[0, 0] = xe[1, 1] - xe[2, 1]
    b[0, 1] = xe[2, 1] - xe[0, 1]
    b[0, 2] = xe[0, 1] - xe[1, 1]

    c = np.zeros((1, 3))
    c[0, 0] = xe[2, 0] - xe[1, 0]
    c[0, 1] = xe[0, 0] - xe[2, 0]
    c[0, 2] = xe[1, 0] - xe[0, 0]

    v12 = xe[1, :] - xe[0, :]
    v13 = xe[2, :] - xe[0, :]
    v23 = xe[2, :] - xe[1, :]

    l = np.zeros((1, 3))
    l[0,0] = np.sqrt(np.dot(v23 ,np.array([v23]).T))
    l[0,1] = np.sqrt(np.dot(v13 , np.array([v13]).T))
    l[0,2] = np.sqrt(np.dot(v12 , np.array([v12]).T))

    mu = np.zeros((1, 3))
    mu[0,0] = (m.pow(l[0,2], 2) - m.pow(l[0,1], 2)) / (m.pow(l[0,0], 2))
    mu[0,1] = (m.pow(l[0,0], 2) - m.pow(l[0,2], 2)) / (m.pow(l[0,1], 2))
    mu[0,2] = (m.pow(l[0,1], 2) - m.pow(l[0,0], 2)) / (m.pow(l[0,2], 2))

    delta = 0.5*(b[0,0]*c[0,1]-b[0,1]*c[0,0])

    A = (1/(2*delta))*np.array([[b[0,0],b[0,1],b[0,2]] , [c[0,0],c[0,1],c[0,2]]])

    Le1=np.eye(3)

    for i in [0,1,2]:
        L = Le1[i, :]
        ddP = np.zeros((9, 6))

        ddP[:, 0] = np.array(
            [[0, 0, 0, 0, 0, 0, 2 * L[1] + L[1] * L[2] * 3 * (1 - mu[0, 2]), L[1] * L[2] * (1 + 3 * mu[0, 0]),
              -L[1] * L[2] * (1 + 3 * mu[0, 1])]])

        ddP[:, 1] = np.array(
            [[0, 0, 0, 0, 0, 0, -L[0] * L[2] * (1 + 3 * mu[0, 2]), 2 * L[2] + L[0] * L[2] * 3 * (1 - mu[0, 0]),
              L[0] * L[2] * (1 + 3 * mu[0, 1])]])

        ddP[:, 2] = np.array([[0, 0, 0, 0, 0, 0, L[0] * L[1] * (1 + 3 * mu[0, 2]), -L[0] * L[1] * (1 + 3 * mu[0, 0]),
                               2 * L[0] + L[0] * L[1] * 3 * (1 - mu[0, 1])]])

        ddP[:, 3] = np.array([[0, 0, 0, 1, 0, 0,
                               2 * L[0] + L[0] * L[2] * 3 * (1 - mu[0, 2]) - L[1] * L[2] * (1 + 3 * mu[0, 2]) + 0.5 * L[
                                   2] ** 2 * (1 + 3 * mu[0, 2]),
                               L[1] * L[2] * 3 * (1 - mu[0, 0]) - 0.5 * L[2] ** 2 * (1 + 3 * mu[0, 0]) + L[0] * L[2] * (
                                       1 + 3 * mu[0, 0]),
                               0.5 * L[2] ** 2 * 3 * (1 - mu[0, 1]) - L[0] * L[2] * (1 + 3 * mu[0, 1]) + L[1] * L[2] * (
                                       1 + 3 * mu[0, 1])]])

        ddP[:, 4] = np.array([[0, 0, 0, 0, 0, 1,
                               L[0] * L[1] * 3 * (1 - mu[0, 2]) - 0.5 * L[1] ** 2 * (1 + 3 * mu[0, 2]) + L[1] * L[2] * (
                                       1 + 3 * mu[0, 2]),
                               0.5 * L[1] ** 2 * 3 * (1 - mu[0, 0]) - L[1] * L[2] * (1 + 3 * mu[0, 0]) + L[0] * L[1] * (
                                       1 + 3 * mu[0, 0]),
                               2 * L[2] + L[1] * L[2] * 3 * (1 - mu[0, 1]) - L[0] * L[1] * (1 + 3 * mu[0, 1]) + 0.5 * L[
                                   1] ** 2 * (1 + 3 * mu[0, 1])]])

        ddP[:, 5] = np.array([[0, 0, 0, 0, 1, 0,
                               0.5 * L[0] ** 2 * 3 * (1 - mu[0, 2]) - L[0] * L[1] * (1 + 3 * mu[0, 2]) + L[0] * L[2] * (
                                       1 + 3 * mu[0, 2]),
                               2 * L[1] + L[0] * L[1] * 3 * (1 - mu[0, 0]) - L[0] * L[2] * (1 + 3 * mu[0, 0]) + 0.5 * L[
                                   0] ** 2 * (1 + 3 * mu[0, 0]),
                               L[0] * L[2] * 3 * (1 - mu[0, 1]) - 0.5 * L[0] ** 2 * (1 + 3 * mu[0, 1]) + L[0] + L[1] * (
                                       1 + 3 * mu[0, 1])]])

        ddN11 = np.array([[ddP[0, 0] - ddP[3, 0] + ddP[5, 0] + 2 * (ddP[6, 0] - ddP[8, 0]),
                           ddP[0, 3] - ddP[3, 3] + ddP[5, 3] + 2 * (ddP[6, 3] - ddP[8, 3]),
                           ddP[0, 4] - ddP[3, 4] + ddP[5, 4] + 2 * (ddP[6, 4] - ddP[8, 4])],
                          [ddP[0, 3] - ddP[3, 3] + ddP[5, 3] + 2 * (ddP[6, 3] - ddP[8, 3]),
                           ddP[0, 1] - ddP[3, 1] + ddP[5, 1] + 2 * (ddP[6, 1] - ddP[8, 1]),
                           ddP[0, 5] - ddP[3, 5] + ddP[5, 5] + 2 * (ddP[6, 5] - ddP[8, 5])],
                          [ddP[0, 4] - ddP[3, 4] + ddP[5, 4] + 2 * (ddP[6, 4] - ddP[8, 4]),
                           ddP[0, 5] - ddP[3, 5] + ddP[5, 5] + 2 * (ddP[6, 5] - ddP[8, 5]),
                           ddP[0, 2] - ddP[3, 2] + ddP[5, 2] + 2 * (ddP[6, 2] - ddP[8, 2])]])

        ddN13 = np.array([[-b[0, 1] * (ddP[8, 0] - ddP[5, 0]) - b[0, 2] * ddP[6, 0],
                           -b[0, 1] * (ddP[8, 3] - ddP[5, 3]) - b[0, 2] * ddP[6, 3],
                           -b[0, 1] * (ddP[8, 4] - ddP[5, 4]) - b[0, 2] * ddP[6, 4]],
                          [-b[0, 1] * (ddP[8, 3] - ddP[5, 3]) - b[0, 2] * ddP[6, 3],
                           -b[0, 1] * (ddP[8, 1] - ddP[5, 1]) - b[0, 2] * ddP[6, 1],
                           -b[0, 1] * (ddP[8, 5] - ddP[5, 5]) - b[0, 2] * ddP[6, 5]],
                          [-b[0, 1] * (ddP[8, 4] - ddP[5, 4]) - b[0, 2] * ddP[6, 4],
                           -b[0, 1] * (ddP[8, 5] - ddP[5, 5]) - b[0, 2] * ddP[6, 5],
                           -b[0, 1] * (ddP[8, 2] - ddP[5, 2]) - b[0, 2] * ddP[6, 2]]])

        ddN12 = -np.array([[-c[0, 1] * (ddP[8, 0] - ddP[5, 0]) - c[0, 2] * ddP[6, 0],
                            -c[0, 1] * (ddP[8, 3] - ddP[5, 3]) - c[0, 2] * ddP[6, 3],
                            -c[0, 1] * (ddP[8, 4] - ddP[5, 4]) - c[0, 2] * ddP[6, 4]],
                           [-c[0, 1] * (ddP[8, 3] - ddP[5, 3]) - c[0, 2] * ddP[6, 3],
                            -c[0, 1] * (ddP[8, 1] - ddP[5, 1]) - c[0, 2] * ddP[6, 1],
                            -c[0, 1] * (ddP[8, 5] - ddP[5, 5]) - c[0, 2] * ddP[6, 5]],
                           [-c[0, 1] * (ddP[8, 4] - ddP[5, 4]) - c[0, 2] * ddP[6, 4],
                            -c[0, 1] * (ddP[8, 5] - ddP[5, 5]) - c[0, 2] * ddP[6, 5],
                            -c[0, 1] * (ddP[8, 2] - ddP[5, 2]) - c[0, 2] * ddP[6, 2]]])

        ddN21 = np.array([[ddP[1, 0] - ddP[4, 0] + ddP[3, 0] + 2 * (ddP[7, 0] - ddP[6, 0]),
                           ddP[1, 3] - ddP[4, 3] + ddP[3, 3] + 2 * (ddP[7, 3] - ddP[6, 3]),
                           ddP[1, 4] - ddP[4, 4] + ddP[3, 4] + 2 * (ddP[7, 4] - ddP[6, 4])],
                          [ddP[1, 3] - ddP[4, 3] + ddP[3, 3] + 2 * (ddP[7, 3] - ddP[6, 3]),
                           ddP[1, 1] - ddP[4, 1] + ddP[3, 1] + 2 * (ddP[7, 1] - ddP[6, 1]),
                           ddP[1, 5] - ddP[4, 5] + ddP[3, 5] + 2 * (ddP[7, 5] - ddP[6, 5])],
                          [ddP[1, 4] - ddP[4, 4] + ddP[3, 4] + 2 * (ddP[7, 4] - ddP[6, 4]),
                           ddP[1, 5] - ddP[4, 5] + ddP[3, 5] + 2 * (ddP[7, 5] - ddP[6, 5]),
                           ddP[1, 2] - ddP[4, 2] + ddP[3, 2] + 2 * (ddP[7, 2] - ddP[6, 2])]])

        ddN23 = np.array([[-b[0, 2] * (ddP[6, 0] - ddP[3, 0]) - b[0, 0] * ddP[7, 0],
                           -b[0, 2] * (ddP[6, 3] - ddP[3, 3]) - b[0, 0] * ddP[7, 3],
                           -b[0, 2] * (ddP[6, 4] - ddP[3, 4]) - b[0, 0] * ddP[7, 4]],
                          [-b[0, 2] * (ddP[6, 3] - ddP[3, 3]) - b[0, 0] * ddP[7, 3],
                           -b[0, 2] * (ddP[6, 1] - ddP[3, 1]) - b[0, 0] * ddP[7, 1],
                           -b[0, 2] * (ddP[6, 5] - ddP[3, 5]) - b[0, 0] * ddP[7, 5]],
                          [-b[0, 2] * (ddP[6, 4] - ddP[3, 4]) - b[0, 0] * ddP[7, 4],
                           -b[0, 2] * (ddP[6, 5] - ddP[3, 5]) - b[0, 0] * ddP[7, 5],
                           -b[0, 2] * (ddP[6, 2] - ddP[3, 2]) - b[0, 0] * ddP[7, 2]]])

        ddN22 = -np.array([[-c[0, 2] * (ddP[6, 0] - ddP[3, 0]) - c[0, 0] * ddP[7, 0],
                            -c[0, 2] * (ddP[6, 3] - ddP[3, 3]) - c[0, 0] * ddP[7, 3],
                            -c[0, 2] * (ddP[6, 4] - ddP[3, 4]) - c[0, 0] * ddP[7, 4]],
                           [-c[0, 2] * (ddP[6, 3] - ddP[3, 3]) - c[0, 0] * ddP[7, 3],
                            -c[0, 2] * (ddP[6, 1] - ddP[3, 1]) - c[0, 0] * ddP[7, 1],
                            -c[0, 2] * (ddP[6, 5] - ddP[3, 5]) - c[0, 0] * ddP[7, 5]],
                           [-c[0, 2] * (ddP[6, 4] - ddP[3, 4]) - c[0, 0] * ddP[7, 4],
                            -c[0, 2] * (ddP[6, 5] - ddP[3, 5]) - c[0, 0] * ddP[7, 5],
                            -c[0, 2] * (ddP[6, 2] - ddP[3, 2]) - c[0, 0] * ddP[7, 2]]])

        ddN31 = np.array([[ddP[2, 0] - ddP[5, 0] + ddP[4, 0] + 2 * (ddP[8, 0] - ddP[7, 0]),
                           ddP[2, 3] - ddP[5, 3] + ddP[4, 3] + 2 * (ddP[8, 3] - ddP[7, 3]),
                           ddP[2, 4] - ddP[5, 4] + ddP[4, 4] + 2 * (ddP[8, 4] - ddP[7, 4])],
                          [ddP[2, 3] - ddP[5, 3] + ddP[4, 3] + 2 * (ddP[8, 3] - ddP[7, 3]),
                           ddP[2, 1] - ddP[5, 1] + ddP[4, 1] + 2 * (ddP[8, 1] - ddP[7, 1]),
                           ddP[2, 5] - ddP[5, 5] + ddP[4, 5] + 2 * (ddP[8, 5] - ddP[7, 5])],
                          [ddP[2, 4] - ddP[5, 4] + ddP[4, 4] + 2 * (ddP[8, 4] - ddP[7, 4]),
                           ddP[2, 5] - ddP[5, 5] + ddP[4, 5] + 2 * (ddP[8, 5] - ddP[7, 5]),
                           ddP[2, 2] - ddP[5, 2] + ddP[4, 2] + 2 * (ddP[8, 2] - ddP[7, 2])]])

        ddN33 = np.array([[-b[0, 0] * (ddP[7, 0] - ddP[4, 0]) - b[0, 1] * ddP[8, 0],
                           -b[0, 0] * (ddP[7, 3] - ddP[4, 3]) - b[0, 1] * ddP[8, 3],
                           -b[0, 0] * (ddP[7, 4] - ddP[4, 4]) - b[0, 1] * ddP[8, 4]],
                          [-b[0, 0] * (ddP[7, 3] - ddP[4, 3]) - b[0, 1] * ddP[8, 3],
                           -b[0, 0] * (ddP[7, 1] - ddP[4, 1]) - b[0, 1] * ddP[8, 1],
                           -b[0, 0] * (ddP[7, 5] - ddP[4, 5]) - b[0, 1] * ddP[8, 5]],
                          [-b[0, 0] * (ddP[7, 4] - ddP[4, 4]) - b[0, 1] * ddP[8, 4],
                           -b[0, 0] * (ddP[7, 5] - ddP[4, 5]) - b[0, 1] * ddP[8, 5],
                           -b[0, 0] * (ddP[7, 2] - ddP[4, 2]) - b[0, 1] * ddP[8, 2]]])

        ddN32 = -np.array([[-c[0, 0] * (ddP[7, 0] - ddP[4, 0]) - c[0, 1] * ddP[8, 0],
                            -c[0, 0] * (ddP[7, 3] - ddP[4, 3]) - c[0, 1] * ddP[8, 3],
                            -c[0, 0] * (ddP[7, 4] - ddP[4, 4]) - c[0, 1] * ddP[8, 4]],
                           [-c[0, 0] * (ddP[7, 3] - ddP[4, 3]) - c[0, 1] * ddP[8, 3],
                            -c[0, 0] * (ddP[7, 1] - ddP[4, 1]) - c[0, 1] * ddP[8, 1],
                            -c[0, 0] * (ddP[7, 5] - ddP[4, 5]) - c[0, 1] * ddP[8, 5]],
                           [-c[0, 0] * (ddP[7, 4] - ddP[4, 4]) - c[0, 1] * ddP[8, 4],
                            -c[0, 0] * (ddP[7, 5] - ddP[4, 5]) - c[0, 1] * ddP[8, 5],
                            -c[0, 0] * (ddP[7, 2] - ddP[4, 2]) - c[0, 1] * ddP[8, 2]]])

        ddN11 = np.dot(np.dot(A, ddN11), A.T)
        ddN12 = np.dot(np.dot(A, ddN12), A.T)
        ddN13 = np.dot(np.dot(A, ddN13), A.T)
        ddN21 = np.dot(np.dot(A, ddN21), A.T)
        ddN22 = np.dot(np.dot(A, ddN22), A.T)
        ddN23 = np.dot(np.dot(A, ddN23), A.T)
        ddN31 = np.dot(np.dot(A, ddN31), A.T)
        ddN32 = np.dot(np.dot(A, ddN32), A.T)
        ddN33 = np.dot(np.dot(A, ddN33), A.T)

        k[3*i:3*(i+1)] = - np.array([[w[T[ie,0]]*ddN11[0,0]+thetax[T[ie,0]]*ddN12[0,0]+thetay[T[ie,0]]*ddN13[0,0]+w[T[ie,1]]*ddN21[0,0]+thetax[T[ie,1]]*ddN22[0,0]+thetay[T[ie,1]]*ddN23[0,0]+w[T[ie,2]]*ddN31[0,0]+thetax[T[ie,2]]*ddN32[0,0]+thetay[T[ie,2]]*ddN33[0,0]],
                                     [w[T[ie,0]]*ddN11[1,1]+thetax[T[ie,0]]*ddN12[1,1]+thetay[T[ie,0]]*ddN13[1,1]+w[T[ie,1]]*ddN21[1,1]+thetax[T[ie,1]]*ddN22[1,1]+thetay[T[ie,1]]*ddN23[1,1]+w[T[ie,2]]*ddN31[1,1]+thetax[T[ie,2]]*ddN32[1,1]+thetay[T[ie,2]]*ddN33[1,1]],
                                     [w[T[ie,0]]*ddN11[0,1]+thetax[T[ie,0]]*ddN12[0,1]+thetay[T[ie,0]]*ddN13[0,1]+w[T[ie,1]]*ddN21[0,1]+thetax[T[ie,1]]*ddN22[0,1]+thetay[T[ie,1]]*ddN23[0,1]+w[T[ie,2]]*ddN31[0,1]+thetax[T[ie,2]]*ddN32[0,1]+thetay[T[ie,2]]*ddN33[0,1]]])

    k[[2, 5, 8]] = 2 * k[[2, 5, 8]]

    return k


def strain_calc_thick(X,T,u,v,w,thetax,thetay,th):


    nT=T.shape[0]
    STRAIN = strain_calc(X,T,u,v)
    thSTRAINxx = np.zeros((nT, 3))
    thSTRAINyy = np.zeros((nT, 3))
    thSTRAINxy = np.zeros((nT, 3))

    if th!=0:
        nT=T.shape[0]

        i=0
        while i<nT :

            k = k_calc(X,T,w,thetax,thetay,i)

            thSTRAINxx[i,0] = STRAIN[i] + th * k[0]
            thSTRAINxx[i,1] = STRAIN[i] + th * k[3]
            thSTRAINxx[i,2] = STRAIN[i] + th * k[6]

            thSTRAINyy[i,0] = STRAIN[i+nT] + th * k[1]
            thSTRAINyy[i,1] = STRAIN[i+nT] + th * k[4]
            thSTRAINyy[i,2] = STRAIN[i+nT] + th * k[7]

            thSTRAINxy[i,0] = STRAIN[i+2*nT] + th * k[2]
            thSTRAINxy[i,1] = STRAIN[i+2*nT] + th * k[5]
            thSTRAINxy[i,2] = STRAIN[i+2*nT] + th * k[8]


            i+=1

    return thSTRAINxx, thSTRAINyy, thSTRAINxy


def stress_calc(X,T,u,v,w,thetax,thetay,th,pos,Qprime,thickness):

    nT=T.shape[0]
    ply=0

    i=0
    while i < pos.shape[0]:
        if pos[i] <= th <= (pos[i]+thickness[i]):
            ply = i
            break

        i += 1


    if th==0:
        stressxx, stressyy, stressxy = np.zeros((nT,1)), np.zeros((nT,1)), np.zeros((nT,1))
        STRAIN = strain_calc(X, T, u, v)

        i=0
        while i<nT:
            index = [i,i+nT,i+2*nT]
            stress = np.dot(Qprime[3*ply:3*ply+3,:],STRAIN[index])

            stressxx[i] = stress[0]
            stressyy[i] = stress[1]
            stressxy[i] = stress[2]

            i+=1
        stressxx, stressyy, stressxy = stressxx[:, 0], stressyy[:, 0], stressxy[:, 0]


    else:

        stressxx, stressyy, stressxy = np.zeros((nT, 3)), np.zeros((nT, 3)), np.zeros((nT, 3))
        thSTRAINxx, thSTRAINyy, thSTRAINxy = strain_calc_thick(X, T, u, v, w, thetax, thetay, th)

        i=0
        while i<nT:
            j=0
            while j<3:

                thSTRAIN = np.array([[thSTRAINxx[i,j]], [thSTRAINyy[i,j]], [thSTRAINxy[i,j]]])
                thstress=np.dot(Qprime[3*ply:3*ply+3,:],thSTRAIN)
                thstress = thstress[:,0]

                stressxx[i,j] = thstress[0]
                stressyy[i,j] = thstress[1]
                stressxy[i,j] = thstress[2]

                j+=1

            i+=1


    return stressxx, stressyy, stressxy


def stress_LT_calc(X,T,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles):
    stressxx, stressyy, stressxy = stress_calc(X, T, u, v, w, thetax, thetay, th, pos, Qprime, thickness)
    nX=X.shape[0]
    nT=T.shape[0]

    i=0
    while i < pos.shape[0]:
        if pos[i] <= th <= (pos[i]+thickness[i]):
            ply = i
            break

        i += 1

    Transform = np.array([[m.cos(angles[ply]) ** 2, m.sin(angles[ply]) ** 2,  2*m.sin(angles[ply]) * m.cos(angles[ply])],
                          [m.sin(angles[ply]) ** 2, m.cos(angles[ply]) ** 2, -2*m.sin(angles[ply]) * m.cos(angles[ply])],
                          [-m.sin(angles[ply]) * m.cos(angles[ply]), 2 * m.sin(angles[ply]) * m.cos(angles[ply]), (m.cos(angles[ply]) ** 2) - m.sin(angles[ply]) ** 2]])

    if th!=0:

        i = 0

        stress_L = np.zeros((nT, 3))
        stress_T = np.zeros((nT, 3))
        stress_LT = np.zeros((nT, 3))

        while i<nT:

            j=0
            while j<3:
                stressinter = np.array([[stressxx[i,j]],[stressyy[i,j]],[stressxy[i,j]]])
                stressiter = np.dot(Transform,stressinter)
                stressiter = stressiter[:,0]

                stress_L[i,j] = stressiter[0]
                stress_T[i,j] = stressiter[1]
                stress_LT[i,j] = stressiter[2]

                j+=1

            i+=1

    else:

        i=0

        stress_L = np.zeros((nT, 1))
        stress_T = np.zeros((nT, 1))
        stress_LT = np.zeros((nT, 1))

        while i<nT:

            stressinter = np.array([[stressxx[i]],[stressyy[i]],[stressxy[i]]])
            stressiter = np.dot(Transform,stressinter)
            stressiter = stressiter[:,0]

            stress_L[i] = stressiter[0]
            stress_T[i] = stressiter[1]
            stress_LT[i] = stressiter[2]

            i += 1

        stress_L, stress_T, stress_LT = stress_L[:,0], stress_T[:,0], stress_LT[:,0]

    return stress_L, stress_T, stress_LT


def strain_LT_calc(X,T,u,v,w,thetax,thetay,th,pos,thickness, angles):
    nX=X.shape[0]
    nT=T.shape[0]

    i=0
    while i < pos.shape[0]:
        if pos[i] <= th <= (pos[i]+thickness[i]):
            ply = i
            break

        i += 1

    Transform = np.array([[m.cos(angles[ply]) ** 2, m.sin(angles[ply]) ** 2, -m.sin(angles[ply]) * m.cos(angles[ply])],
                          [m.sin(angles[ply]) ** 2, m.cos(angles[ply]) ** 2,  m.sin(angles[ply]) * m.cos(angles[ply])],
                          [2*m.sin(angles[ply]) * m.cos(angles[ply]), -2 * m.sin(angles[ply]) * m.cos(angles[ply]), (m.cos(angles[ply]) ** 2) - m.sin(angles[ply]) ** 2]])

    if th!=0:

        thSTRAINxx, thSTRAINyy, thSTRAINxy = strain_calc_thick(X, T, u, v, w, thetax, thetay, th)

        strain_L = np.zeros((nT, 3))
        strain_T = np.zeros((nT, 3))
        strain_LT = np.zeros((nT, 3))

        i = 0
        while i<nT:



            j=0
            while j<3:


                straininter = np.array([[thSTRAINxx[i,j]],[thSTRAINyy[i,j]],[thSTRAINxy[i,j]]])
                strainiter = np.dot(Transform,straininter)
                strainiter = strainiter[:,0]

                strain_L[i,j] = strainiter[0]
                strain_T[i,j] = strainiter[1]
                strain_LT[i,j] = strainiter[2]

                j+=1

            i+=1

    else:

        STRAIN = strain_calc(X, T, u, v)
        STRAIN = STRAIN[:,0]

        strain_L = np.zeros((nT, 1))
        strain_T = np.zeros((nT, 1))
        strain_LT = np.zeros((nT, 1))

        i=0

        while i<nT:



            straininter = np.array([[STRAIN[i]],[STRAIN[nT+i]],[STRAIN[2*nT+i]]])
            strainiter = np.dot(Transform,straininter)
            strainiter = strainiter[:,0]

            strain_L[i] = strainiter[0]
            strain_T[i] = strainiter[1]
            strain_LT[i] = strainiter[2]

            i += 1

        strain_L, strain_T, strain_LT = strain_L[:,0], strain_T[:,0], strain_LT[:,0]

    return strain_L, strain_T, strain_LT

def Hoffman(SLLt, SLLc, STTt, STTc, SLT, X,T,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles):
    stress_L, stress_T, stress_LT = stress_LT_calc(X,T,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles)
    nT= T.shape[0]
    nX=X.shape[0]

    if th==0:
        Hoffman_stress = np.zeros((nT, 1))

        i = 0
        while i < nT:
            Hoffman_stress[i] = ((stress_L[i] ** 2) / (SLLt * SLLc)) + ((stress_T[i] ** 2) / (STTt * STTc)) - (
                        (stress_T[i] * stress_L[i]) / (SLLt * SLLc)) + ((SLLc - SLLt) / (SLLt * SLLc)) * stress_L[i] + (
                                            (STTc - STTt) / (STTt * STTc)) * stress_T[i] + (stress_LT[i] ** 2) / (
                                            SLT ** 2)
            i += 1

        Hoffman_stress = Hoffman_stress[:, 0]

    else:
        Hoffman_stress = np.zeros((nT, 3))

        i = 0
        while i < nT:
            j=0
            while j<3:
                Hoffman_stress[i,j] = ((stress_L[i,j] ** 2) / (SLLt * SLLc)) + ((stress_T[i,j] ** 2) / (STTt * STTc)) - (
                        (stress_T[i,j] * stress_L[i,j]) / (SLLt * SLLc)) + ((SLLc - SLLt) / (SLLt * SLLc)) * stress_L[i,j] + (
                                            (STTc - STTt) / (STTt * STTc)) * stress_T[i,j] + (stress_LT[i,j] ** 2) / (
                                            SLT ** 2)

                j+=1
            i += 1


    return Hoffman_stress


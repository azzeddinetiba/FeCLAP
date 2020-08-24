# -*-coding:Latin-1 -*

import numpy as np
import math as m

def ElemMat(X,T,ie,gp,Wgauss,K):

    xe=X[T[ie,:],:]
    J = np.array([(xe[1,:]-xe[0,:]).T,(xe[2,:]-xe[0,:]).T]).T
    Ke=np.zeros((18,18))

    b = np.zeros((1, 3))
    b[0,0] = xe[1, 1] - xe[2, 1]
    b[0,1] = xe[2, 1] - xe[0, 1]
    b[0,2] = xe[0, 1] - xe[1, 1]

    c = np.zeros((1, 3))
    c[0,0] = xe[2, 0] - xe[1, 0]
    c[0,1] = xe[0, 0] - xe[2, 0]
    c[0,2] = xe[1, 0] - xe[0, 0]

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

    dN1dx = b[0,0] / (2 * delta)
    dN1dy = c[0,0] / (2 * delta)
    dN2dx = b[0,1] / (2 * delta)
    dN2dy = c[0,1] / (2 * delta)
    dN3dx = b[0,2] / (2 * delta)
    dN3dy = c[0,2] / (2 * delta)

    L1=gp

    ii=0
    while ii < Wgauss.shape[0] :
     L = L1[ii,:]
     ddP = np.zeros((9, 6))

     ddP[:, 0] = np.array([[0, 0, 0, 0, 0, 0, 2 * L[1] + L[1] * L[2] * 3 * (1 - mu[0,2]), L[1] * L[2] * (1 + 3 * mu[0,0]),
                            -L[1] * L[2] * (1 + 3 * mu[0,1])]])

     ddP[:, 1] = np.array([[0, 0, 0, 0, 0, 0, -L[0] * L[2] * (1 + 3 * mu[0,2]), 2 * L[2] + L[0] * L[2] * 3 * (1 - mu[0,0]),
                            L[0] * L[2] * (1 + 3 * mu[0,1])]])

     ddP[:, 2] = np.array([[0, 0, 0, 0, 0, 0, L[0] * L[1] * (1 + 3 * mu[0,2]), -L[0] * L[1] * (1 + 3 * mu[0,0]),
                            2 * L[0] + L[0] * L[1] * 3 * (1 - mu[0,1])]])

     ddP[:, 3] = np.array([[0, 0, 0, 1, 0, 0,
                            2 * L[0] + L[0] * L[2] * 3 * (1 - mu[0,2]) - L[1] * L[2] * (1 + 3 * mu[0,2]) + 0.5 * L[
                                2] ** 2 * (1 + 3 * mu[0,2]),
                            L[1] * L[2] * 3 * (1 - mu[0,0]) - 0.5 * L[2] ** 2 * (1 + 3 * mu[0,0]) + L[0] * L[2] * (
                                        1 + 3 * mu[0,0]),
                            0.5 * L[2] ** 2 * 3 * (1 - mu[0,1]) - L[0] * L[2] * (1 + 3 * mu[0,1]) + L[1] * L[2] * (
                                        1 + 3 * mu[0,1])]])

     ddP[:, 4] = np.array([[0, 0, 0, 0, 0, 1,
                            L[0] * L[1] * 3 * (1 - mu[0,2]) - 0.5 * L[1] ** 2 * (1 + 3 * mu[0,2]) + L[1] * L[2] * (
                                        1 + 3 * mu[0,2]),
                            0.5 * L[1] ** 2 * 3 * (1 - mu[0,0]) - L[1] * L[2] * (1 + 3 * mu[0,0]) + L[0] * L[1] * (
                                        1 + 3 * mu[0,0]),
                            2 * L[2] + L[1] * L[2] * 3 * (1 - mu[0,1]) - L[0] * L[1] * (1 + 3 * mu[0,1]) + 0.5 * L[
                                1] ** 2 * (1 + 3 * mu[0,1])]])

     ddP[:, 5] = np.array([[0, 0, 0, 0, 1, 0,
                            0.5 * L[0] ** 2 * 3 * (1 - mu[0,2]) - L[0] * L[1] * (1 + 3 * mu[0,2]) + L[0] * L[2] * (
                                        1 + 3 * mu[0,2]),
                            2 * L[1] + L[0] * L[1] * 3 * (1 - mu[0,0]) - L[0] * L[2] * (1 + 3 * mu[0,0]) + 0.5 * L[
                                0] ** 2 * (1 + 3 * mu[0,0]),
                            L[0] * L[2] * 3 * (1 - mu[0,1]) - 0.5 * L[0] ** 2 * (1 + 3 * mu[0,1]) + L[0] + L[1] * (
                                        1 + 3 * mu[0,1])]])

     ddN11 = np.array([[ddP[0, 0] - ddP[3, 0] + ddP[5, 0] + 2 * (ddP[6, 0] - ddP[8, 0]),
                        ddP[0, 3] - ddP[3, 3] + ddP[5, 3] + 2 * (ddP[6, 3] - ddP[8, 3]),
                        ddP[0, 4] - ddP[3, 4] + ddP[5, 4] + 2 * (ddP[6, 4] - ddP[8, 4])],
                       [ddP[0, 3] - ddP[3, 3] + ddP[5, 3] + 2 * (ddP[6, 3] - ddP[8, 3]),
                        ddP[0, 1] - ddP[3, 1] + ddP[5, 1] + 2 * (ddP[6, 1] - ddP[8, 1]),
                        ddP[0, 5] - ddP[3, 5] + ddP[5, 5] + 2 * (ddP[6, 5] - ddP[8, 5])],
                       [ddP[0, 4] - ddP[3, 4] + ddP[5, 4] + 2 * (ddP[6, 4] - ddP[8, 4]),
                        ddP[0, 5] - ddP[3, 5] + ddP[5, 5] + 2 * (ddP[6, 5] - ddP[8, 5]),
                        ddP[0, 2] - ddP[3, 2] + ddP[5, 2] + 2 * (ddP[6, 2] - ddP[8, 2])]])

     ddN13 = np.array([[-b[0,1] * (ddP[8, 0] - ddP[5, 0]) - b[0,2] * ddP[6, 0],
                        -b[0,1] * (ddP[8, 3] - ddP[5, 3]) - b[0,2] * ddP[6, 3],
                        -b[0,1] * (ddP[8, 4] - ddP[5, 4]) - b[0,2] * ddP[6, 4]],
                       [-b[0,1] * (ddP[8, 3] - ddP[5, 3]) - b[0,2] * ddP[6, 3],
                        -b[0,1] * (ddP[8, 1] - ddP[5, 1]) - b[0,2] * ddP[6, 1],
                        -b[0,1] * (ddP[8, 5] - ddP[5, 5]) - b[0,2] * ddP[6, 5]],
                       [-b[0,1] * (ddP[8, 4] - ddP[5, 4]) - b[0,2] * ddP[6, 4],
                        -b[0,1] * (ddP[8, 5] - ddP[5, 5]) - b[0,2] * ddP[6, 5],
                        -b[0,1] * (ddP[8, 2] - ddP[5, 2]) - b[0,2] * ddP[6, 2]]])

     ddN12 = -np.array([[-c[0,1] * (ddP[8, 0] - ddP[5, 0]) - c[0,2] * ddP[6, 0],
                         -c[0,1] * (ddP[8, 3] - ddP[5, 3]) - c[0,2] * ddP[6, 3],
                         -c[0,1] * (ddP[8, 4] - ddP[5, 4]) - c[0,2] * ddP[6, 4]],
                        [-c[0,1] * (ddP[8, 3] - ddP[5, 3]) - c[0,2] * ddP[6, 3],
                         -c[0,1] * (ddP[8, 1] - ddP[5, 1]) - c[0,2] * ddP[6, 1],
                         -c[0,1] * (ddP[8, 5] - ddP[5, 5]) - c[0,2] * ddP[6, 5]],
                        [-c[0,1] * (ddP[8, 4] - ddP[5, 4]) - c[0,2] * ddP[6, 4],
                         -c[0,1] * (ddP[8, 5] - ddP[5, 5]) - c[0,2] * ddP[6, 5],
                         -c[0,1] * (ddP[8, 2] - ddP[5, 2]) - c[0,2] * ddP[6, 2]]])

     ddN21 = np.array([[ddP[1, 0] - ddP[4, 0] + ddP[3, 0] + 2 * (ddP[7, 0] - ddP[6, 0]),
                        ddP[1, 3] - ddP[4, 3] + ddP[3, 3] + 2 * (ddP[7, 3] - ddP[6, 3]),
                        ddP[1, 4] - ddP[4, 4] + ddP[3, 4] + 2 * (ddP[7, 4] - ddP[6, 4])],
                       [ddP[1, 3] - ddP[4, 3] + ddP[3, 3] + 2 * (ddP[7, 3] - ddP[6, 3]),
                        ddP[1, 1] - ddP[4, 1] + ddP[3, 1] + 2 * (ddP[7, 1] - ddP[6, 1]),
                        ddP[1, 5] - ddP[4, 5] + ddP[3, 5] + 2 * (ddP[7, 5] - ddP[6, 5])],
                       [ddP[1, 4] - ddP[4, 4] + ddP[3, 4] + 2 * (ddP[7, 4] - ddP[6, 4]),
                        ddP[1, 5] - ddP[4, 5] + ddP[3, 5] + 2 * (ddP[7, 5] - ddP[6, 5]),
                        ddP[1, 2] - ddP[4, 2] + ddP[3, 2] + 2 * (ddP[7, 2] - ddP[6, 2])]])

     ddN23 = np.array([[-b[0,2] * (ddP[6, 0] - ddP[3, 0]) - b[0,0] * ddP[7, 0],
                        -b[0,2] * (ddP[6, 3] - ddP[3, 3]) - b[0,0] * ddP[7, 3],
                        -b[0,2] * (ddP[6, 4] - ddP[3, 4]) - b[0,0] * ddP[7, 4]],
                       [-b[0,2] * (ddP[6, 3] - ddP[3, 3]) - b[0,0] * ddP[7, 3],
                        -b[0,2] * (ddP[6, 1] - ddP[3, 1]) - b[0,0] * ddP[7, 1],
                        -b[0,2] * (ddP[6, 5] - ddP[3, 5]) - b[0,0] * ddP[7, 5]],
                       [-b[0,2] * (ddP[6, 4] - ddP[3, 4]) - b[0,0] * ddP[7, 4],
                        -b[0,2] * (ddP[6, 5] - ddP[3, 5]) - b[0,0] * ddP[7, 5],
                        -b[0,2] * (ddP[6, 2] - ddP[3, 2]) - b[0,0] * ddP[7, 2]]])

     ddN22 = -np.array([[-c[0,2] * (ddP[6, 0] - ddP[3, 0]) - c[0,0] * ddP[7, 0],
                         -c[0,2] * (ddP[6, 3] - ddP[3, 3]) - c[0,0] * ddP[7, 3],
                         -c[0,2] * (ddP[6, 4] - ddP[3, 4]) - c[0,0] * ddP[7, 4]],
                        [-c[0,2] * (ddP[6, 3] - ddP[3, 3]) - c[0,0] * ddP[7, 3],
                         -c[0,2] * (ddP[6, 1] - ddP[3, 1]) - c[0,0] * ddP[7, 1],
                         -c[0,2] * (ddP[6, 5] - ddP[3, 5]) - c[0,0] * ddP[7, 5]],
                        [-c[0,2] * (ddP[6, 4] - ddP[3, 4]) - c[0,0] * ddP[7, 4],
                         -c[0,2] * (ddP[6, 5] - ddP[3, 5]) - c[0,0] * ddP[7, 5],
                         -c[0,2] * (ddP[6, 2] - ddP[3, 2]) - c[0,0] * ddP[7, 2]]])

     ddN31 = np.array([[ddP[2, 0] - ddP[5, 0] + ddP[4, 0] + 2 * (ddP[8, 0] - ddP[7, 0]),
                        ddP[2, 3] - ddP[5, 3] + ddP[4, 3] + 2 * (ddP[8, 3] - ddP[7, 3]),
                        ddP[2, 4] - ddP[5, 4] + ddP[4, 4] + 2 * (ddP[8, 4] - ddP[7, 4])],
                       [ddP[2, 3] - ddP[5, 3] + ddP[4, 3] + 2 * (ddP[8, 3] - ddP[7, 3]),
                        ddP[2, 1] - ddP[5, 1] + ddP[4, 1] + 2 * (ddP[8, 1] - ddP[7, 1]),
                        ddP[2, 5] - ddP[5, 5] + ddP[4, 5] + 2 * (ddP[8, 5] - ddP[7, 5])],
                       [ddP[2, 4] - ddP[5, 4] + ddP[4, 4] + 2 * (ddP[8, 4] - ddP[7, 4]),
                        ddP[2, 5] - ddP[5, 5] + ddP[4, 5] + 2 * (ddP[8, 5] - ddP[7, 5]),
                        ddP[2, 2] - ddP[5, 2] + ddP[4, 2] + 2 * (ddP[8, 2] - ddP[7, 2])]])

     ddN33 = np.array([[-b[0,0] * (ddP[7, 0] - ddP[4, 0]) - b[0,1] * ddP[8, 0],
                        -b[0,0] * (ddP[7, 3] - ddP[4, 3]) - b[0,1] * ddP[8, 3],
                        -b[0,0] * (ddP[7, 4] - ddP[4, 4]) - b[0,1] * ddP[8, 4]],
                       [-b[0,0] * (ddP[7, 3] - ddP[4, 3]) - b[0,1] * ddP[8, 3],
                        -b[0,0] * (ddP[7, 1] - ddP[4, 1]) - b[0,1] * ddP[8, 1],
                        -b[0,0] * (ddP[7, 5] - ddP[4, 5]) - b[0,1] * ddP[8, 5]],
                       [-b[0,0] * (ddP[7, 4] - ddP[4, 4]) - b[0,1] * ddP[8, 4],
                        -b[0,0] * (ddP[7, 5] - ddP[4, 5]) - b[0,1] * ddP[8, 5],
                        -b[0,0] * (ddP[7, 2] - ddP[4, 2]) - b[0,1] * ddP[8, 2]]])

     ddN32 = -np.array([[-c[0,0] * (ddP[7, 0] - ddP[4, 0]) - c[0,1] * ddP[8, 0],
                         -c[0,0] * (ddP[7, 3] - ddP[4, 3]) - c[0,1] * ddP[8, 3],
                         -c[0,0] * (ddP[7, 4] - ddP[4, 4]) - c[0,1] * ddP[8, 4]],
                        [-c[0,0] * (ddP[7, 3] - ddP[4, 3]) - c[0,1] * ddP[8, 3],
                         -c[0,0] * (ddP[7, 1] - ddP[4, 1]) - c[0,1] * ddP[8, 1],
                         -c[0,0] * (ddP[7, 5] - ddP[4, 5]) - c[0,1] * ddP[8, 5]],
                        [-c[0,0] * (ddP[7, 4] - ddP[4, 4]) - c[0,1] * ddP[8, 4],
                         -c[0,0] * (ddP[7, 5] - ddP[4, 5]) - c[0,1] * ddP[8, 5],
                         -c[0,0] * (ddP[7, 2] - ddP[4, 2]) - c[0,1] * ddP[8, 2]]])

     ddN11 = np.dot(np.dot(A, ddN11), A.T)
     ddN12 = np.dot(np.dot(A, ddN12), A.T)
     ddN13 = np.dot(np.dot(A, ddN13), A.T)
     ddN21 = np.dot(np.dot(A, ddN21), A.T)
     ddN22 = np.dot(np.dot(A, ddN22), A.T)
     ddN23 = np.dot(np.dot(A, ddN23), A.T)
     ddN31 = np.dot(np.dot(A, ddN31), A.T)
     ddN32 = np.dot(np.dot(A, ddN32), A.T)
     ddN33 = np.dot(np.dot(A, ddN33), A.T)


     B=np.array([[dN1dx ,0     ,0            ,0            ,0            ,0 ,dN2dx ,0     ,0            ,0            ,0             ,0 ,dN3dx ,0     ,0            ,0            ,0           ,0],
                 [0     ,dN1dy ,0            ,0            ,0            ,0 ,0     ,dN2dy ,0            ,0            ,0             ,0 ,0     ,dN3dy ,0            ,0            ,0           ,0],
                 [dN1dy ,dN1dx ,0            ,0            ,0            ,0 ,dN2dy ,dN2dx ,0            ,0            ,0             ,0 ,dN3dy ,dN3dx ,0            ,0            ,0           ,0],
                 [0     ,0     ,ddN11[0,0]   ,ddN12[0,0]   ,ddN13[0,0]   ,0 ,0     ,0     ,ddN21[0,0]   ,ddN22[0,0]   ,ddN23[0,0]    ,0 ,0     ,0     ,ddN31[0,0]   ,ddN32[0,0]   ,ddN33[0,0]  ,0],
                 [0     ,0     ,ddN11[1,1]   ,ddN12[1,1]   ,ddN13[1,1]   ,0 ,0     ,0     ,ddN21[1,1]   ,ddN22[1,1]   ,ddN23[1,1]    ,0 ,0     ,0     ,ddN31[1,1]   ,ddN32[1,1]   ,ddN33[1,1]  ,0],
                 [0     ,0     ,2*ddN11[0,1] ,2*ddN12[0,1] ,2*ddN13[0,1] ,0 ,0     ,0     ,2*ddN21[0,1] ,2*ddN22[0,1] ,2*ddN23[0,1]  ,0 ,0     ,0     ,2*ddN31[0,1] ,2*ddN32[0,1] ,2*ddN33[0,1] ,0]])




     Ke += Wgauss[ii] * np.linalg.det(J)*np.dot(np.dot(B.T, K), B)
     ii+=1

    return Ke

def ElemMassMat(X, T, ie, gp, Wgauss, pho, thickness):

        pliesnum = pho.size
        pliesthick = np.zeros((pliesnum, 1))

        xe = X[T[ie, :], :]
        J = np.array([(xe[1, :] - xe[0, :]).T, (xe[2, :] - xe[0, :]).T]).T
        Me = np.zeros((18, 18))

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
        l[0, 0] = np.sqrt(np.dot(v23, np.array([v23]).T))
        l[0, 1] = np.sqrt(np.dot(v13, np.array([v13]).T))
        l[0, 2] = np.sqrt(np.dot(v12, np.array([v12]).T))

        mu = np.zeros((1, 3))
        mu[0, 0] = (m.pow(l[0, 2], 2) - m.pow(l[0, 1], 2)) / (m.pow(l[0, 0], 2))
        mu[0, 1] = (m.pow(l[0, 0], 2) - m.pow(l[0, 2], 2)) / (m.pow(l[0, 1], 2))
        mu[0, 2] = (m.pow(l[0, 1], 2) - m.pow(l[0, 0], 2)) / (m.pow(l[0, 2], 2))

        delta = 0.5 * (b[0, 0] * c[0, 1] - b[0, 1] * c[0, 0])

        A = (1 / (2 * delta)) * np.array([[b[0, 0], b[0, 1], b[0, 2]], [c[0, 0], c[0, 1], c[0, 2]]])

        ii = 0
        while ii < Wgauss.shape[0]:
            L = gp[ii, :]
            N1 = L[0]
            N2 = L[1]
            N3 = L[2]

            P = np.array([[N1],
                          [N2],
                          [N3],
                          [N1 * N2],
                          [N2 * N3],
                          [N3 * N1],
                          [N2 * N1 ** 2 + 0.5 * N1 * N2 * N3 * (
                                      3 * (1 - mu[0, 2]) * N1 - (1 + 3 * mu[0, 2]) * N2 + N3 * (1 + 3 * mu[0, 2]))],
                          [N3 * N2 ** 2 + 0.5 * N1 * N2 * N3 * (
                                      3 * (1 - mu[0, 0]) * N2 - (1 + 3 * mu[0, 0]) * N3 + N1 * (1 + 3 * mu[0, 0]))],
                          [N1 * N3 ** 2 + 0.5 * N1 * N2 * N3 * (
                                      3 * (1 - mu[0, 1]) * N3 - (1 + 3 * mu[0, 1]) * N1 + N2 * (1 + 3 * mu[0, 1]))]])

            P = P[:, 0]

            N = np.array([[N1],
                          [N1],
                          [P[0] - P[3] + P[5] + 2 * (P[6] - P[8])],
                          [- (-c[0, 1] * (P[8] - P[5]) - c[0, 2] * P[6])],
                          [- b[0, 1] * (P[8] - P[5]) - b[0, 2] * P[6]],
                          [0],
                          [N2],
                          [N2],
                          [P[1] - P[4] + P[3] + 2 * (P[7] - P[8])],
                          [- (-c[0, 2] * (P[6] - P[3]) - c[0, 0] * P[7])],
                          [- b[0, 2] * (P[6] - P[3]) - b[0, 0] * P[7]],
                          [0],
                          [N3],
                          [N3],
                          [P[2] - P[5] + P[4] + 2 * (P[8] - P[7])],
                          [- (-c[0, 0] * (P[7] - P[4]) - c[0, 1] * P[8])],
                          [- b[0, 0] * (P[7] - P[4]) - b[0, 1] * P[8]],
                          [0]])

            N = N[:, 0]

            B = np.array([[N[0], 0, 0, 0, 0, 0, N[6], 0, 0, 0, 0, 0, N[12], 0, 0, 0, 0, 0],
                          [0, N[1], 0, 0, 0, 0, 0, N[7], 0, 0, 0, 0, 0, N[13], 0, 0, 0, 0],
                          [0, 0, N[2], N[3], N[4], 0, 0, 0, N[8], N[9], N[10], 0, 0, 0, N[14], N[15], N[16], 0]])

            Me += Wgauss[ii] * np.linalg.det(J) * np.dot(B.T, B)
            ii += 1

        elemMass = 0
        jj = 0
        while jj < pliesnum:
            interm = pho[jj] * thickness[jj]
            elemMass += Me * interm[0]
            jj += 1

        return elemMass
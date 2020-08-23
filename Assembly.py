# -*-coding:Latin-1 -*
import os
import numpy as np
import math as m
import matplotlib.pyplot as plt
import cv2
import pylab
from scipy import linalg as alg
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy import sparse
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv, jn_zeros

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

def Assembly2D(X,T,f,g,h,Wgauss,gp,Ngauss,Klaw,pho,thickness,analysis_type):

    Nn=6*X.shape[0]
    Nt=T.shape[0]

    K=np.zeros((Nn,Nn))
    F=np.zeros((Nn,1))
    M=np.zeros((Nn,Nn))

    ie=0
    while ie < Nt :
     Tie=T[ie,:]+1
     Tie1=np.concatenate((np.arange(6*Tie[0]-5,6*Tie[0]+1),np.arange(6*Tie[1]-5,6*Tie[1]+1),np.arange(6*Tie[2]-5,6*Tie[2]+1)),0)
     Ke = ElemMat(X, T, ie, gp, Wgauss, Klaw)
     Fe = SMelem(f, g, h, X, T, ie, Ngauss, Wgauss)
     K[np.ix_(Tie1-1,Tie1-1)]+=Ke
     F[Tie1-1,:]+=Fe
     if analysis_type[0,0] !=1:
         Me = ElemMassMat(X, T, ie, gp, Wgauss, pho, thickness)
         M[np.ix_(Tie1 - 1, Tie1 - 1)] += Me

     ie+=1

    return K,M,F


def BCAssembly(X,T,b,F,j,fixedx):

    FBC=np.zeros((b.size,1))
    ie=0
    while ie<b.size-1:
        FEBC=SMBC(F,X,T,ie,b,j,fixedx)
        for ii in [1,2]:
            I=ie+ii-1
            FBC[I,0]+=FEBC[ii-1,0]

        ie+=1

    return FBC


def BCMAssembly(X,T,b,F,j,fixedx,xy):

    FBCM=np.zeros((b.size,1))
    ie=0
    while ie<b.size-1:
        FEBCM=SMBCM(F,X,T,ie,b,j,fixedx,xy)
        for ii in [1,2]:
            I=ie+ii-1
            FBCM[I,0]+=FEBCM[ii-1,0]

        ie+=1

    return FBCM


def GAUSS(N):

    if N==1:
     points = 0.000000
     weights = 2.000
    elif N==2:
     points = np.array([-0.5773502691896257,0.5773502691896257])
     weights = np.array([1.000000000000000,1.0000000000000000])
    elif N==3:
     points = np.array([0.0000000000000000,-0.7745966692414834,0.7745966692414834])
     weights = np.array([0.8888888888888888,0.5555555555555556,0.5555555555555556])
    elif N==4:
     points = np.array([-0.3399810435848563,0.3399810435848563,- 0.8611363115940526,0.8611363115940526])
     weights = np.array([0.6521451548625461,0.6521451548625461,0.3478548451374538,0.3478548451374538])
    elif N==5:
     points = np.array([0.0000000000000000,- 0.5384693101056831,0.5384693101056831,- 0.9061798459386640,0.9061798459386640])
     weights = np.array([0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891])
    elif N==6:
     points = np.array([-0.6612093864662645,0.6612093864662645,- 0.2386191860831969,0.2386191860831969,- 0.9324695142031521,0.9324695142031521])
     weights = np.array([0.3607615730481386,0.3607615730481386,0.4679139345726910,0.4679139345726910,0.1713244923791704,0.1713244923791704])

    return points,weights


def Gauss3n(n):
    if n == 1:
        gp = np.array([ 1/3,1/3,1/3 ])
        gw = 1
    elif n == 3:
        gp = np.array([[1/2,1/2,0], [1/2,0,1/2] , [0,1/2,1/2]])
        gw = np.array([1/3,1/3,1/3])
    elif n == 4:
        gp = np.array([[1/3 ,1/3 ,1/3] , [0.6, 0.2, 0.2] , [0.2,0.6,0.2] , [0.2,0.2,0.6]])
        gw = np.array([-27/48,25/48,25/48,25/48])
    elif n == 7:
        alpha1 = 0.0597158717
        beta1 = 0.4701420641
        alpha2 = 0.7974269853
        beta2 = 0.1012865073
        gp = np.array([[1/3,1/3,1/3],[alpha1,beta1,beta1],[beta1,alpha1,beta1],[beta1,beta1,alpha1],[alpha2,beta2,beta2],[beta2,alpha2,beta2],[beta2,beta2,alpha2]])
        gw = np.array([ 0.2250000000 , 0.1323941527 , 0.1323941527 , 0.1323941527,0.1259391805 , 0.1259391805 , 0.1259391805])


    return gp,gw


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


def FEM(f,g,h,NX1,NY1,NX2,NY2,NX3,NY3,NX4,NY4,MY1,MXY1,MX2,MXY2,MY3,MXY3,MX4,MXY4,boundaryconditions,ENFRCDS,X,T,b,Ngauss,Klaw,box,pointload,NODALLOAD,pho,thickness,analysis_type,transient):

    #U = np.zeros((1,X.shape[0]))

    ind=0
    while ind<4:
        i=0
        while i<b.shape[1]:
            if b[ind,i]==0:
                if ind==0:
                 b1=b[ind,np.arange(0,i)]
                elif ind==1:
                 b2 = b[ind, np.arange(0, i)]
                elif ind==2:
                 b3 = b[ind, np.arange(0, i)]
                elif ind==3:
                 b4 = b[ind, np.arange(0, i)]
                break
            elif b[ind,i]!=0 and i==b.shape[1]-1:
                if ind==0:
                 b1=b[ind,np.arange(0,i+1)]
                elif ind==1:
                 b2 = b[ind, np.arange(0, i+1)]
                elif ind==2:
                 b3 = b[ind, np.arange(0, i+1)]
                elif ind==3:
                 b4 = b[ind, np.arange(0, i+1)]
                break
            i+=1
        ind+=1

    Xgauss, Wgauss = Quadrature(1, Ngauss)
    gp= Gauss3n(Ngauss)
    gp = gp[0]

    K, M, F = Assembly2D(X, T, f, g, h, Wgauss, gp, Ngauss, Klaw, pho,thickness,analysis_type)

    ii=0
    while ii< b1.size:
     min = ii
     kk=ii+1
     while kk< b1.size:
      if X[b1[kk]-1, 1] < X[b1[min]-1, 1]:
        min = kk

      kk+=1
     rempl = b1[min]
     b1[min] = b1[ii]
     b1[ii] = rempl
     ii+=1

    ii = 0
    while ii < b2.size:
         min = ii
         kk = ii + 1
         while kk < b2.size:
             if X[b2[kk]-1, 0] < X[b2[min]-1, 0]:
                 min = kk

             kk += 1
         rempl = b2[min]
         b2[min] = b2[ii]
         b2[ii] = rempl
         ii += 1

    ii = 0
    while ii < b3.size:
         min = ii
         kk = ii + 1
         while kk < b3.size:
             if X[b3[kk]-1, 1] < X[b3[min]-1, 1]:
                 min = kk

             kk += 1
         rempl = b3[min]
         b3[min] = b3[ii]
         b3[ii] = rempl
         ii += 1


    ii = 0
    while ii < b4.size:
     min = ii
     kk = ii + 1
     while kk < b4.size:
      if X[b4[kk]-1, 0] < X[b4[min]-1, 0]:
         min = kk
      kk += 1

     rempl = b4[min]
     b4[min] = b4[ii]
     b4[ii] = rempl
     ii += 1

     ENFRCDS1 = np.array([ENFRCDS[:, np.arange(0,5)]])
     ENFRCDS2 = np.array([ENFRCDS[:, np.arange(5,10)]])
     ENFRCDS1 = ENFRCDS1[0]
     ENFRCDS2 = ENFRCDS2[0]
     x1 = box[0,0]
     y1 = box[0,1]
     x2 = box[1,0]
     y2 = box[1,1]

    border1 = np.zeros((1, 6*b1.size))
    ii=0
    while ii<b1.size:
        border1[0,np.arange(6 * ii , 6 * ii+6)]=np.arange(6 * b1[ii] - 5, 6 * b1[ii]+1)
        ii+=1
    border1=border1.astype(int)
    border1 = border1[0]

    border2 = np.zeros((1, 6 * b2.size))
    ii = 0
    while ii < b2.size:
        border2[0,np.arange(6 * ii , 6 * ii+6)]=np.arange(6 * b2[ii] - 5, 6 * b2[ii]+1)
        ii += 1
    border2 = border2.astype(int)
    border2 = border2[0]

    border3 = np.zeros((1, 6 * b3.size))
    ii = 0
    while ii < b3.size:
        border3[0,np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b3[ii] - 5, 6 * b3[ii]+1)
        ii += 1
    border3 = border3.astype(int)
    border3 = border3[0]

    border4 = np.zeros((1, 6 * b4.size))
    ii = 0
    while ii < b4.size:
     border4[0,np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b4[ii] - 5, 6 * b4[ii]+1)
     ii += 1
    border4 = border4.astype(int)
    border4 = border4[0]

    border_size = max(border1.shape[0], border2.shape[0], border3.shape[0], border4.shape[0])
    everyborder = np.zeros((4, border_size))
    everyborder[0, :] = np.concatenate((border1, np.zeros((1, border_size - border1.shape[0]))), axis=None)
    everyborder[1, :] = np.concatenate((border2, np.zeros((1, border_size - border2.shape[0]))), axis=None)
    everyborder[2, :] = np.concatenate((border3, np.zeros((1, border_size - border3.shape[0]))), axis=None)
    everyborder[3, :] = np.concatenate((border4, np.zeros((1, border_size - border4.shape[0]))), axis=None)
    everyborder = everyborder.astype(int)



    if boundaryconditions[0] == 1:
      F[border1[np.arange(0,border1.size,6)]-1] += BCAssembly(X, T, b1, NX1, 2, x1)
      F[border1[np.arange(1,border1.size,6)]-1] += BCAssembly(X, T, b1, NY1, 2, x1)
    elif boundaryconditions[0] == 2:
      K[border1-1,:]=0
      K[:, border1-1]=0
      M[border1-1,:]=0
      M[:, border1-1]=0
      F[border1-1] = 0
      K[np.ix_(border1-1, border1-1)] = np.eye(border1.size)
      M[np.ix_(border1-1, border1-1)] = np.eye(border1.size)


    elif boundaryconditions[0] == 3:
      F[border1[np.arange(3,border1.size,6)]-1,:] += BCMAssembly(X, T, b1, MY1, 2, x1, 1)
      F[border1[np.arange(4,border1.size,6)]-1,:] += BCMAssembly(X, T, b1, MXY1, 2, x1, 2)
    elif boundaryconditions[0] == 4:
      srch=np.nonzero(ENFRCDS2[0,:])
      for c in srch[0]:
       K[border1[np.arange(c,border1.size,6)]-1,:]=0
       M[border1[np.arange(c,border1.size,6)]-1,:]=0
       F[border1[np.arange(c,border1.size,6)]-1]=ENFRCDS1[0, c]
       #K[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)]=np.eye(border1[np.arange(c,border1.size,6)].size)
       K[np.ix_(border1[np.arange(c, border1.size, 6)] - 1, border1[np.arange(c, border1.size, 6)] - 1)] -= np.diag(np.diag(K[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)])) - np.eye(border1[np.arange(c,border1.size,6)].size)
       M[np.ix_(border1[np.arange(c, border1.size, 6)] - 1, border1[np.arange(c, border1.size, 6)] - 1)] -= np.diag(np.diag(M[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)])) - np.eye(border1[np.arange(c,border1.size,6)].size)

    elif boundaryconditions[0] == 5:
        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        M[border1[::6] - 1, :] = 0
        M[:, border1[::6] - 1] = 0
        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)

        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        M[border1[1::6] - 1, :] = 0
        M[:, border1[1::6] - 1] = 0
        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)

        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        M[border1[2::6] - 1, :] = 0
        M[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)

        K[border1[3::6] - 1, :] = 0
        K[:, border1[3::6] - 1] = 0
        M[border1[3::6] - 1, :] = 0
        M[:, border1[3::6] - 1] = 0
        F[border1[3::6] - 1] = 0
        K[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)
        M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

    elif boundaryconditions[0] == 6:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        M[border1[1::6] - 1, :] = 0
        M[:, border1[1::6] - 1] = 0
        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)

        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        M[border1[2::6] - 1, :] = 0
        M[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)

        K[border1[3::6] - 1, :] = 0
        K[:, border1[3::6] - 1] = 0
        M[border1[3::6] - 1, :] = 0
        M[:, border1[3::6] - 1] = 0
        F[border1[3::6] - 1] = 0
        K[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)
        M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

    elif boundaryconditions[0] == 7:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        M[border1[1::6] - 1, :] = 0
        M[:, border1[1::6] - 1] = 0
        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)

        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        M[border1[::6] - 1, :] = 0
        M[:, border1[::6] - 1] = 0
        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)

        K[border1[4::6] - 1, :] = 0
        K[:, border1[4::6] - 1] = 0
        M[border1[4::6] - 1, :] = 0
        M[:, border1[4::6] - 1] = 0
        F[border1[4::6] - 1] = 0
        K[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)
        M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

    elif boundaryconditions[0] == 8:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        M[border1[1::6] - 1, :] = 0
        M[:, border1[1::6] - 1] = 0
        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)

        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        M[border1[::6] - 1, :] = 0
        M[:, border1[::6] - 1] = 0
        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)

        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        M[border1[2::6] - 1, :] = 0
        M[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)

        K[border1[4::6] - 1, :] = 0
        K[:, border1[4::6] - 1] = 0
        M[border1[4::6] - 1, :] = 0
        M[:, border1[4::6] - 1] = 0
        F[border1[4::6] - 1] = 0
        K[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)
        M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)





    if boundaryconditions[1] == 1:
      F[border2[np.arange(0,border2.size,6)]-1,:] += BCAssembly(X, T, b2, NX2, 1, y1)
      F[border2[np.arange(1,border2.size,6)]-1,:] += BCAssembly(X, T, b2, NY2, 1, y1)
      if boundaryconditions[0] == 2:
       K[border2[np.arange(0,6)]-1,:]=0
       K[:, border2[np.arange(0,6)]-1]=0
       M[border2[np.arange(0,6)]-1,:]=0
       M[:, border2[np.arange(0,6)]-1]=0
       F[border2[np.arange(0,6)]-1] = 0
       K[np.ix_(border2[np.arange(0,6)]-1, border2[np.arange(0,6)]-1)] = np.eye(6)
       M[np.ix_(border2[np.arange(0,6)]-1, border2[np.arange(0,6)]-1)] = np.eye(6)
    elif boundaryconditions[1] == 2:
      K[border2-1,:] = 0
      K[:, border2-1] = 0
      M[border2-1,:] = 0
      M[:, border2-1] = 0
      F[border2-1] = 0
      K[np.ix_(border2-1, border2-1)] = np.eye(border2.size)
      M[np.ix_(border2-1, border2-1)] = np.eye(border2.size)


    elif boundaryconditions[1] == 3:
     F[border2[np.arange(4,border2.size,6)]-1]+= BCMAssembly(X, T, b2, MX2, 1, y1, 2)
     F[border2[np.arange(3,border2.size,6)]-1]+= BCMAssembly(X, T, b2, MXY2, 1, y1, 1)
     if boundaryconditions[0] == 2:
       K[border1-1,:]=0
       K[:, border1-1]=0
       F[border1-1] = 0
       K[np.ix_(border1-1, border1-1)] = np.eye(border1.size)
       M[border1-1,:]=0
       M[:, border1-1]=0
       M[np.ix_(border1-1, border1-1)] = np.eye(border1.size)
    elif boundaryconditions[1] == 4:
      srch=np.nonzero(ENFRCDS2[1,:])
      for c in srch[0]:
       K[border2[np.arange(c,border2.size,6)]-1,:]=0
       F[border2[np.arange(c,border2.size,6)]-1,:]=ENFRCDS1[1, c]
       #K[np.ix_(border2[np.arange(c,border2.size,6)]-1, border2[np.arange(c,border2.size,6)]-1)]=np.eye(border2[np.arange(c,border2.size,6)].size)
       K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)] -= np.diag(np.diag(K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)]))  - np.eye(border2[np.arange(c,border2.size,6)].size)
       M[border2[np.arange(c,border2.size,6)]-1,:]=0
       M[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)] -= np.diag(np.diag(K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)]))  - np.eye(border2[np.arange(c,border2.size,6)].size)


    elif boundaryconditions[1] == 5:
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        M[border2[::6] - 1, :] = 0
        M[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)

        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0
        M[border2[1::6] - 1, :] = 0
        M[:, border2[1::6] - 1] = 0
        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)

        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        M[border2[2::6] - 1, :] = 0
        M[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)

        K[border2[4::6] - 1, :] = 0
        K[:, border2[4::6] - 1] = 0
        M[border2[4::6] - 1, :] = 0
        M[:, border2[4::6] - 1] = 0
        F[border2[4::6] - 1] = 0
        K[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)
        M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

    elif boundaryconditions[1] == 6:
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        M[border2[::6] - 1, :] = 0
        M[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)

        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        M[border2[2::6] - 1, :] = 0
        M[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)

        K[border2[4::6] - 1, :] = 0
        K[:, border2[4::6] - 1] = 0
        M[border2[4::6] - 1, :] = 0
        M[:, border2[4::6] - 1] = 0
        F[border2[4::6] - 1] = 0
        K[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)
        M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

    elif boundaryconditions[1] == 7:
        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0
        M[border2[1::6] - 1, :] = 0
        M[:, border2[1::6] - 1] = 0
        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)

        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        M[border2[::6] - 1, :] = 0
        M[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)

        K[border2[3::6] - 1, :] = 0
        K[:, border2[3::6] - 1] = 0
        M[border2[3::6] - 1, :] = 0
        M[:, border2[3::6] - 1] = 0
        F[border2[3::6] - 1] = 0
        K[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)
        M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

    elif boundaryconditions[1] == 8:
        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0
        M[border2[1::6] - 1, :] = 0
        M[:, border2[1::6] - 1] = 0
        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)

        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        M[border2[::6] - 1, :] = 0
        M[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)

        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        M[border2[2::6] - 1, :] = 0
        M[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)

        K[border2[3::6] - 1, :] = 0
        K[:, border2[3::6] - 1] = 0
        M[border2[3::6] - 1, :] = 0
        M[:, border2[3::6] - 1] = 0
        F[border2[3::6] - 1] = 0
        K[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)
        M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)





    if boundaryconditions[2] == 1:
      F[border3[np.arange(0,border3.size,6)]-1] += BCAssembly(X, T, b3, NX3, 2, x2)
      F[border3[np.arange(1,border3.size,6)]-1] += BCAssembly(X, T, b3, NY3, 2, x2)
      if boundaryconditions[1] == 2:
       K[border3[np.arange(0,6)]-1,:] = 0
       K[:, border3[np.arange(0,6)]-1] = 0
       F[border3[np.arange(0,6)]-1] = 0
       K[np.ix_(border3[np.arange(0,6)]-1, border3[np.arange(0,6)]-1)] = np.eye(6)
    elif boundaryconditions[2] == 2:
      K[border3-1,:]=0
      M[border3-1,:]=0
      K[:, border3-1]=0
      M[:, border3-1]=0
      F[border3-1] = 0
      K[np.ix_(border3-1, border3-1)] = np.eye(border3.size)
      M[np.ix_(border3-1, border3-1)] = np.eye(border3.size)


    elif boundaryconditions[2] == 3:
      F[border3[np.arange(3,border3.size,6)]-1] += BCMAssembly(X, T, b3, MY3, 2, x2, 1)
      F[border3[np.arange(4,border3.size,6)]-1] += BCMAssembly(X, T, b3, MXY3, 2, x2, 2)
      if boundaryconditions[1] == 2:
       K[border2-1,:]=0
       K[:, border2-1]=0
       F[border2-1] = 0
       K[np.ix_(border2-1, border2-1)] = np.eye(border2.size)
    elif boundaryconditions[2] == 4:
      srch=np.nonzero(ENFRCDS2[2,:])
      for c in srch[0]:
       K[border3[np.arange(c,border3.size,6)]-1,:]=0
       F[border3[np.arange(c,border3.size,6)]-1]=ENFRCDS1[2, c]
       #K[np.ix_(border3[np.arange(c,border3.size,6)]-1, border3[np.arange(c,border3.size,6)]-1)]=np.eye(border3[np.arange(c,border3.size,6)].size)
       K[np.ix_(border3[np.arange(c,border3.size,6)]-1, border3[np.arange(c,border3.size,6)]-1)] -= np.diag(np.diag(K[np.ix_(border3[np.arange(c,border3.size,6)]-1, border3[np.arange(c,border3.size,6)]-1)])) - np.eye(border3[np.arange(c,border3.size,6)].size)

    elif boundaryconditions[2] == 5:
        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0
        M[border3[::6] - 1, :] = 0
        M[:, border3[::6] - 1] = 0
        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)

        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        M[border3[1::6] - 1, :] = 0
        M[:, border3[1::6] - 1] = 0
        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)

        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        M[border3[2::6] - 1, :] = 0
        M[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)

        K[border3[3::6] - 1, :] = 0
        K[:, border3[3::6] - 1] = 0
        M[border3[3::6] - 1, :] = 0
        M[:, border3[3::6] - 1] = 0
        F[border3[3::6] - 1] = 0
        K[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)
        M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)


    elif boundaryconditions[2] == 6:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        M[border3[1::6] - 1, :] = 0
        M[:, border3[1::6] - 1] = 0
        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)

        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        M[border3[2::6] - 1, :] = 0
        M[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)

        K[border3[3::6] - 1, :] = 0
        K[:, border3[3::6] - 1] = 0
        M[border3[3::6] - 1, :] = 0
        M[:, border3[3::6] - 1] = 0
        F[border3[3::6] - 1] = 0
        K[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)
        M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

    elif boundaryconditions[2] == 7:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        M[border3[1::6] - 1, :] = 0
        M[:, border3[1::6] - 1] = 0
        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)

        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0
        M[border3[::6] - 1, :] = 0
        M[:, border3[::6] - 1] = 0
        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)

        K[border3[4::6] - 1, :] = 0
        K[:, border3[4::6] - 1] = 0
        M[border3[4::6] - 1, :] = 0
        M[:, border3[4::6] - 1] = 0
        F[border3[4::6] - 1] = 0
        K[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)
        M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

    elif boundaryconditions[2] == 8:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        M[border3[1::6] - 1, :] = 0
        M[:, border3[1::6] - 1] = 0
        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)

        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0
        M[border3[::6] - 1, :] = 0
        M[:, border3[::6] - 1] = 0
        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)

        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        M[border3[2::6] - 1, :] = 0
        M[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)

        K[border3[4::6] - 1, :] = 0
        K[:, border3[4::6] - 1] = 0
        M[border3[4::6] - 1, :] = 0
        M[:, border3[4::6] - 1] = 0
        F[border3[4::6] - 1] = 0
        K[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)
        M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)





    if boundaryconditions[3] == 1:
      F[border4[np.arange(0,border4.size,6)]-1] += BCAssembly(X, T, b4, NX4, 1, y2)
      F[border4[np.arange(1,border4.size,6)]-1] += BCAssembly(X, T, b4, NY4, 1, y2)
      if boundaryconditions[2] == 2:
       sz4=border4.size
       K[border4[np.arange(sz4-1,sz4-7,-1)]-1,:]=0
       K[:, border4[np.arange(sz4-1,sz4-7,-1)]-1]=0
       F[border4[np.arange(sz4-1,sz4-7,-1)]-1] = 0
       K[np.ix_(border4[np.arange(sz4-1,sz4-7,-1)]-1, border4[np.arange(sz4-1,sz4-7,-1)]-1)] = np.eye(6)
      if boundaryconditions[0] == 2:
       K[border1[np.arange(0,6)] - 1, :] = 0
       K[:, border1[np.arange(0,6)]-1] = 0
       F[border1[np.arange(0,6)]-1] = 0
       K[np.ix_(border1[np.arange(0,6)] - 1, border1[np.arange(0,6)] - 1)] = np.eye(6)
    elif boundaryconditions[3] == 2:
      K[border4-1,:]=0
      M[border4-1,:]=0
      M[:, border4-1]=0
      K[:, border4-1]=0
      F[border4-1] = 0
      K[np.ix_(border4-1, border4-1)] = np.eye(border4.size)
      M[np.ix_(border4-1, border4-1)] = np.eye(border4.size)


    elif boundaryconditions[3] == 3:
      F[border4[np.arange(4,border4.size,6)]-1]+= BCMAssembly(X, T, b4, MX4, 1, y2, 2)
      F[border4[np.arange(3,border4.size,6)]-1]+= BCMAssembly(X, T, b4, MXY4, 1, y2, 1)
      if boundaryconditions[2] == 2:
       K[border3[np.arange(0,6)]-1,:]=0
       K[:, border3[np.arange(0,6)]-1]=0
       F[border3[np.arange(0,6)]-1] = 0
       K[np.ix_(border3[np.arange(0,6)]-1, border3[np.arange(0,6)]-1)] = np.eye(6)
      if boundaryconditions[0] == 2:
       K[border1[np.arange(0,6)]-1,:]=0
       K[:, border1[np.arange(0,6)]-1]=0
       F[border1[np.arange(0,6)]-1] = 0
       K[np.ix_(border1[np.arange(0,6)]-1, border1[np.arange(0,6)]-1)] = np.eye(6)
    elif boundaryconditions[3] == 4:
      srch=np.nonzero(ENFRCDS2[3,:])
      for c in srch[0]:
       K[border4[np.arange(c,border4.size,6)]-1,:]=0
       F[border4[np.arange(c,border4.size,6)]-1,:]=ENFRCDS1[3, c]
       #K[np.ix_(border4[np.arange(c,border4.size,6)]-1, border4[np.arange(c,border4.size,6)]-1)]=np.eye(border4[np.arange(c,border4.size,6)].size)
       K[np.ix_(border4[np.arange(c, border4.size, 6)] - 1, border4[np.arange(c, border4.size, 6)] - 1)] -= np.diag(np.diag(K[np.ix_(border4[np.arange(c, border4.size, 6)] - 1, border4[np.arange(c, border4.size, 6)] - 1)])) - np.eye(border4[np.arange(c,border4.size,6)].size)

    elif boundaryconditions[3] == 5:
        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0
        M[border4[::6] - 1, :] = 0
        M[:, border4[::6] - 1] = 0
        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0
        M[border4[1::6] - 1, :] = 0
        M[:, border4[1::6] - 1] = 0
        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
        M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0
        M[border4[2::6] - 1, :] = 0
        M[:, border4[2::6] - 1] = 0
        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
        M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[4::6] - 1, :] = 0
        K[:, border4[4::6] - 1] = 0
        M[border4[4::6] - 1, :] = 0
        M[:, border4[4::6] - 1] = 0
        F[border4[4::6] - 1] = 0
        K[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)
        M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

    elif boundaryconditions[3] == 6:
        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0
        M[border4[::6] - 1, :] = 0
        M[:, border4[::6] - 1] = 0
        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0
        M[border4[2::6] - 1, :] = 0
        M[:, border4[2::6] - 1] = 0
        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
        M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[4::6] - 1, :] = 0
        K[:, border4[4::6] - 1] = 0
        M[border4[4::6] - 1, :] = 0
        M[:, border4[4::6] - 1] = 0
        F[border4[4::6] - 1] = 0
        K[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)
        M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

    elif boundaryconditions[3] == 7:
        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0
        M[border4[1::6] - 1, :] = 0
        M[:, border4[1::6] - 1] = 0
        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
        M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0
        M[border4[::6] - 1, :] = 0
        M[:, border4[::6] - 1] = 0
        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[3::6] - 1, :] = 0
        K[:, border4[3::6] - 1] = 0
        M[border4[3::6] - 1, :] = 0
        M[:, border4[3::6] - 1] = 0
        F[border4[3::6] - 1] = 0
        K[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)
        M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

    elif boundaryconditions[3] == 8:
        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0
        M[border4[1::6] - 1, :] = 0
        M[:, border4[1::6] - 1] = 0
        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
        M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0
        M[border4[::6] - 1, :] = 0
        M[:, border4[::6] - 1] = 0
        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0
        M[border4[2::6] - 1, :] = 0
        M[:, border4[2::6] - 1] = 0
        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
        M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[3::6] - 1, :] = 0
        K[:, border4[3::6] - 1] = 0
        M[border4[3::6] - 1, :] = 0
        M[:, border4[3::6] - 1] = 0
        F[border4[3::6] - 1] = 0
        K[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)
        M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)






    K[np.ix_(np.arange(5,K.shape[0],6), np.arange(5,K.shape[0],6))]=np.eye(int(K.shape[0] / 6))
    M[np.ix_(np.arange(5,M.shape[0],6), np.arange(5,M.shape[0],6))]=np.eye(int(M.shape[0] / 6))


    fixed_borders = np.array([])
    fixed_index = np.where(boundaryconditions == 2)[0]
    fixed_borders = everyborder[fixed_index]
    fixed_borders = fixed_borders.flatten()
    #fixed_borders = fixed_borders[np.nonzero(fixed_borders)]

    fixed_index2 = np.where(boundaryconditions == 5)[0]
    i=0
    while i<fixed_index2.shape[0]:
        fixed_borders_interm = everyborder[fixed_index2[i]]
        if i%2!=0:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[2::6],fixed_borders_interm[4::6]))
        else:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[2::6],fixed_borders_interm[3::6]))

        fixed_borders = np.concatenate((fixed_borders,fixed_borders_interm_int))
        i+=1

    fixed_index2 = np.where(boundaryconditions == 6)[0]
    i=0
    while i<fixed_index2.shape[0]:
        fixed_borders_interm = everyborder[fixed_index2[i]]
        if i%2!=0:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[1::6],fixed_borders_interm[2::6],fixed_borders_interm[4::6]))
        else:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[2::6],fixed_borders_interm[3::6]))

        fixed_borders = np.concatenate((fixed_borders,fixed_borders_interm_int))
        i+=1

    fixed_index2 = np.where(boundaryconditions == 7)[0]
    i=0
    while i<fixed_index2.shape[0]:
        fixed_borders_interm = everyborder[fixed_index2[i]]
        if i%2!=0:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[3::6]))
        else:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[4::6]))

        fixed_borders = np.concatenate((fixed_borders,fixed_borders_interm_int))
        i+=1

    fixed_index2 = np.where(boundaryconditions == 8)[0]
    i=0
    while i<fixed_index2.shape[0]:
        fixed_borders_interm = everyborder[fixed_index2[i]]
        if i%2!=0:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[2::6],fixed_borders_interm[3::6]))
        else:
         fixed_borders_interm_int = np.concatenate((fixed_borders_interm[0::6],fixed_borders_interm[1::6],fixed_borders_interm[2::6],fixed_borders_interm[4::6]))

        fixed_borders = np.concatenate((fixed_borders,fixed_borders_interm_int))
        i+=1

    Msize=M.shape[0]
    fixed_borders = np.concatenate((fixed_borders, np.arange(5,Msize,6)+1))
    fixed_borders = fixed_borders[np.nonzero(fixed_borders)]

    i=0
    if not (NODALLOAD.size == 0 ):
        while i < pointload.shape[0]:
            while (np.any(np.isin(fixed_borders-1, 6 * (pointload[i]+1) - 6)) and NODALLOAD[i,0]!=0) or (np.any(np.isin(fixed_borders-1, 6 * (pointload[i]+1) - 5)) and NODALLOAD[i,1]!=0) or (np.any(np.isin(fixed_borders-1, 6 * (pointload[i]+1) - 4)) and NODALLOAD[i,2]!=0):
                XY = np.zeros((1,2))
                print ('The mesh used can t allow a nodal load this close to the constrained border, please change the location of the load : ')
                print('Load application point coordinates : [x y] ')
                for j in np.arange(0, 2):
                    XY[0,j] = float(input())
                pointload = -1
                distload = 10000
                ii=0
                while ii<X.shape[0]:
                    if m.sqrt(((X[ii,0]-XY[0,0])**2)+((X[ii,1]-XY[0,1])**2)) <= distload:
                        distload = m.sqrt(((X[ii,0]-XY[0,0])**2)+((X[ii,1]-XY[0,1])**2))
                        pointload[i] = ii
                    ii += 1
            F[6 * (pointload[i] + 1) - 6] += NODALLOAD[i, 0]
            F[6 * (pointload[i] + 1) - 5] += NODALLOAD[i, 1]
            F[6 * (pointload[i] + 1) - 4] += NODALLOAD[i, 2]
            i+=1



    Mb = np.delete(M, fixed_borders - 1, 0)
    Mb = np.delete(Mb, fixed_borders - 1, 1)
    Kb = np.delete(K, fixed_borders - 1, 0)
    Kb = np.delete(Kb, fixed_borders - 1, 1)


    #Ksize=Kb.shape[0]
    #Msize=Mb.shape[0]
    #Mb = np.delete(Mb, np.arange(5,Msize,6), 0)
    #Mb = np.delete(Mb, np.arange(5,Msize,6), 1)
    #Kb = np.delete(Kb, np.arange(5,Ksize,6), 0)
    #Kb = np.delete(Kb, np.arange(5,Ksize,6), 1)


    fixed_borders = np.unique(fixed_borders)
    xxx = np.arange(0,M.shape[0])
    i=0
    while i<fixed_borders.shape[0]:
     if i==0:
      index = np.argwhere(xxx == fixed_borders[i]-1)
     else:
      index = np.argwhere(yyy == fixed_borders[i] - 1)
     if i==0 :
         yyy = np.delete(xxx, index)
     else:
         yyy = np.delete(yyy, index)

     i+=1


    #if i==0:
    # modal_indexes = np.delete(xxx, np.arange(5,xxx.shape[0],6), 0)
    #else:
    # yyy = np.delete(yyy, np.arange(5,yyy.shape[0],6), 0)
    modal_indexes = yyy


    Fb=F.T
    Fb=Fb[0]

    if analysis_type[0,0] == 1:

        if analysis_type[0,1] == 1:
            U = np.linalg.solve(K, Fb)
            return K, F, U
        else:

            U, p_strain = plastic_analysis (K, Fb)
            return U, p_strain



    elif analysis_type[0,0] == 3:

        modes_number = Mb.shape[0]#int(input('Number of modes extracted ? '))
        w, modes = eig(Kb,Mb)

        modes = modes.real
        indices = np.argsort(w)
        modes = modes[:,indices]
        w = np.sort(w)

        freq = (np.sqrt(w))*(1/(2 * m.pi))
        freq = freq.real
        freq = freq[0:modes_number]
        modes = modes[:,0:modes_number]



        return Kb, Mb, freq, modes, modal_indexes

    else:


        delta_T = transient['time_step']

        tf = transient['final_instant']

        n_iter = tf/delta_T

        #F = np.dot(F,np.array([np.concatenate((np.linspace(0,1,int(n_iter/6)),np.linspace(1,0,int(n_iter/6)),np.zeros((1,int(n_iter)-2*int(n_iter/6)))[0]))]))
        F = np.dot(F, np.ones((1,int(n_iter))))

        if transient['scheme']==1:
            A = (delta_T**2)*K + M
            A2 = np.eye(M.shape[0]) + np.dot(np.linalg.inv(A),M)
            U2 = (delta_T**2)*np.dot(np.linalg.inv(A2),np.dot(np.linalg.inv(A),F[:,1]))

            SOL = np.zeros((M.shape[0],int(n_iter)))
            SOL[:,1] = U2.T

            i=2
            while i<n_iter:

                U = np.dot(np.linalg.inv(A),(((delta_T**2)*F[:,i])-(np.dot(M,SOL[:,i-2])).T+(2*np.dot(M,SOL[:,i-1])).T))
                SOL[:,i] = U.T
                i+=1

        elif transient['scheme']==3:

            n_dof = M.shape[0]
            SOL = np.zeros((n_dof,int(n_iter)))

            x = float(input('Initial displacement : ? [m]'))
            xdot = float(input('Initial velocity : ? [m/s]'))
            x = x*np.ones((n_dof,1))
            xdot = xdot*np.ones((n_dof,1))
            xtwodots = np.linalg.solve(M,F[:,0]-(np.dot(K,x)))

            x = x.T[0]
            xdot = xdot.T[0]
            xtwodots = xtwodots.T[0]


            x1 = x - delta_T*xdot + 0.5 * (delta_T**2) * xtwodots

            mat = np.linalg.inv((1/(delta_T**2))*M)
            A = (2/(delta_T**2))*M - K
            B = -(1/(delta_T**2))*M

            SOL[:, 0] = x
            SOL[:, 1] = x1

            i=2
            while i<int(n_iter):

                SOL[:,i] = np.dot(mat,(np.dot(A,SOL[:,i-1])+np.dot(B,SOL[:,i-2])+F[:,i]))
                i+=1

        else:

            beta = float(input('Beta  :'))
            gamma = float(input('gamma :'))

            n_dof = M.shape[0]
            SOL = np.zeros((3*n_dof,int(n_iter)))


            x = float(input('Initial displacement : ? [m]'))
            xdot = float(input('Initial velocity : ? [m/s]'))
            x = x*np.ones((n_dof,1))
            xdot = xdot*np.ones((n_dof,1))
            xtwodots = -np.dot(np.linalg.inv(M), np.dot(K, x) - F[:, 0])

            x = x.T[0]
            xdot = xdot.T[0]
            xtwodots = xtwodots.T[0]


            SOL[0:n_dof, i - 1] = x
            SOL[n_dof:2 * n_dof, i - 1] = xdot
            SOL[2 * n_dof: 3 * n_dof, i - 1] = xtwodots

            mat= np.linalg.inv((1/(beta*delta_T**2))*M+K)
            inv_M = np.linalg.inv(M)

            i=1
            while i<int(n_iter):


                #deltax = np.dot(np.linalg.inv((1/(beta*delta_T**2))*M+K),(F[:,i]-F[:,i-1])+np.dot(M,(1/(beta*delta_T))*xdot+(1/(beta*2))*xtwodots))
                #deltax = np.linalg.solve((1/(beta*delta_T**2))*M+K,(F[:,i]-F[:,i-1])+np.dot(M,(1/(beta*delta_T))*xdot+(1/(beta*2))*xtwodots))
                deltax = np.dot(mat,(F[:,i]-F[:,i-1])+np.dot(M,(1/(beta*delta_T))*xdot+(1/(beta*2))*xtwodots))

                x += deltax
                xdot += (deltax * gamma / (beta * delta_T)) - (gamma / beta) * xdot + delta_T * (
                            1 - gamma / (2 * beta)) * xtwodots

                #xtwodots = -np.dot(np.linalg.inv(M),np.dot(K,x)-F[:,i])
                #xtwodots = -np.linalg.solve(M,np.dot(K,x)-F[:,i])
                xtwodots = -np.dot(inv_M, np.dot(K, x) - F[:, i])

                SOL[0:n_dof, i] = x
                SOL[n_dof:2 * n_dof, i] = xdot
                SOL[2 * n_dof: 3 * n_dof, i] = xtwodots



                i += 1


        return K, F, SOL


def Quadrature(elem, Ngauss):
    if elem == 0:
     if Ngauss == 4:
      pos1 = 1 / m.sqrt[2]
      Xgauss = np.array([[-pos1, - pos1],
                  [pos1, - pos1],
                  [pos1,    pos1],
                  [- pos1,    pos1]])
      Wgauss = np.array([1,1,1,1])
     elif Ngauss == 9:
      pos1 = m.sqrt(3 / 5)
      Xgauss = np.array([[-pos1, - pos1],
                         [0, - pos1],
                         [pos1, - pos1],
                         [- pos1,     0],
                         [0,     0],
                         [pos1,     0],
                         [- pos1,    pos1],
                         [0,     pos1],
                         [pos1,    pos1]])
      pg1 = 5 / 9
      pg2 = 8 / 9
      pg3 = pg1
      Wgauss = np.array([pg1 * pg1, pg2 * pg1, pg3 * pg1, pg1 * pg2, pg2 * pg2, pg3 * pg2, pg1 * pg3, pg2 * pg3, pg3 * pg3])

    elif elem == 1:
     if Ngauss == 1:
      pos1 = 1 / 3
      Xgauss = np.array([pos1,   pos1])
      Wgauss = 1 / 2
     elif Ngauss == 3:
      pos1 = 1 / 2
      Xgauss = np.array([[pos1,   pos1],
                         [0,      pos1],
                         [pos1,   0]])
      pes1 = 1 / 6
      Wgauss = np.array([pes1,   pes1,   pes1])
     elif Ngauss == 4:
      Xgauss = np.array([[1 / 3,   1 / 3],
                         [0.6,   0.2],
                         [0.2,   0.6],
                         [0.2,   0.2]])
      Wgauss = np.array([-27 / 96,   25 / 96,   25 / 96,   25 / 96])
     elif Ngauss == 7:
      a = 0.101286507323456338800987361915123
      b = 0.470142064105115089770441209513447
      Xgauss = np.array([[a,       a],
                         [a,      1 - 2 * a],
                         [1 - 2 * a,   a],
                         [b,       b],
                         [b,       1 - 2 * b],
                         [1 - 2 * b,   b],
                         [1 / 3,     1 / 3]])
      Wgauss = 0.5 * np.array([0.2250000000, 0.1323941527, 0.1323941527, 0.1323941527, 0.1259391805, 0.1259391805, 0.1259391805])

    return Xgauss, Wgauss


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


def SMelem(f,g,h,X,T,ie,Ngauss,Wgauss):

    Fe=np.zeros((18,1))
    gp,gw=Gauss3n(Ngauss)
    gw=Wgauss
    xe = X[T[ie, :], :]
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
    l[0,0] = np.sqrt(np.dot(v23 , np.array([v23]).T))
    l[0,1] = np.sqrt(np.dot(v13 , np.array([v13]).T))
    l[0,2] = np.sqrt(np.dot(v12 , np.array([v12]).T))

    mu = np.zeros((1, 3))
    mu[0,0] = (m.pow(l[0,2], 2) - m.pow(l[0,1], 2)) / (m.pow(l[0,0], 2))
    mu[0,1] = (m.pow(l[0,0], 2) - m.pow(l[0,2], 2)) / (m.pow(l[0,1], 2))
    mu[0,2] = (m.pow(l[0,1], 2) - m.pow(l[0,0], 2)) / (m.pow(l[0,2], 2))

    ii=0
    while ii<Ngauss:
        L=gp[ii,:]
        N1=L[0]
        N2=L[1]
        N3=L[2]

        P = np.array([[N1],
        [N2],
        [N3],
        [N1 * N2],
        [N2 * N3],
        [N3 * N1],
        [N2 * N1 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,2]) * N1 - (1 + 3 * mu[0,2]) * N2 + N3 * (1 + 3 * mu[0,2]))],
        [N3 * N2 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,0]) * N2 - (1 + 3 * mu[0,0]) * N3 + N1 * (1 + 3 * mu[0,0]))],
        [N1 * N3 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,1]) * N3 - (1 + 3 * mu[0,1]) * N1 + N2 * (1 + 3 * mu[0,1]))]])

        N = np.array([[P[0] - P[3] + P[5] + 2 * (P[6] - P[8])],
             [- (-c[0,1] * (P[8] - P[5]) - c[0,2] * P[6])],
             [- b[0,1] * (P[8] - P[5]) - b[0,2] * P[6]],
             [P[1] - P[4] + P[3] + 2 * (P[7] - P[8])],
             [- (-c[0,2] * (P[6] - P[3]) - c[0,0] * P[7])],
             [- b[0,2] * (P[6] - P[3]) - b[0,0] * P[7]],
             [P[2] - P[5] + P[4] + 2 * (P[8] - P[7])],
             [- (-c[0,0] * (P[7] - P[4]) - c[0,1] * P[8])],
             [- b[0,0] * (P[7] - P[4]) - b[0,1] * P[8]]])

        J = np.array([[xe[0, 0],xe[1, 0],xe[2, 0]],[xe[0,1], xe[1,1], xe[2,1]],[1,1,1]])

        Fe[0,0] += gw[ii] * np.linalg.det(J) * N1 * g(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[1,0] += gw[ii] * np.linalg.det(J) * N1 * h(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[6,0] += gw[ii] * np.linalg.det(J) * N2 * g(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[7,0] += gw[ii] * np.linalg.det(J) * N2 * h(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[12,0] += gw[ii] * np.linalg.det(J) * N3 * g(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[13,0] += gw[ii] * np.linalg.det(J) * N3 * h(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])

        Fe[2,0] += gw[ii] * np.linalg.det(J) * N[0] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[3,0] += gw[ii] * np.linalg.det(J) * N[1] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[4,0] += gw[ii] * np.linalg.det(J) * N[2] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])

        Fe[8,0] += gw[ii] * np.linalg.det(J) * N[3] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[9,0] += gw[ii] * np.linalg.det(J) * N[4] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[10,0] += gw[ii] * np.linalg.det(J) * N[5] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])

        Fe[14,0] += gw[ii] * np.linalg.det(J) * N[6] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[15,0] += gw[ii] * np.linalg.det(J) * N[7] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])
        Fe[16,0] += gw[ii] * np.linalg.det(J) * N[8] * f(L[0] * xe[0,0] + L[1] * xe[1,0] + L[2] * xe[2,0],L[0] * xe[0,1] + L[1] * xe[1,1] + L[2] * xe[2, 1])

        ii+=1
    return Fe


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


def mesh(x1, x2, y1, y2, h, radius=0, xh=0, yh=0):
    import distmesh as dm
    import matplotlib.pyplot as plt

    if radius==0:
     fd = lambda p: dm.drectangle(p, x1, x2, y1, y2)
     fig = plt.figure()
     p, t = dm.distmesh2d(fd, dm.huniform, h, (x1, y1, x2, y2), [(x1, y1), (x1, y2), (x2, y1), (x2, y2)])
    else:
     fd = lambda p: dm.ddiff(dm.drectangle(p, x1, x2, y1, y2),dm.dcircle(p, xh, yh, radius))
     fh = lambda p: h + h*6 * dm.dcircle(p, xh, yh, radius)

     fig = plt.figure()

     p, t = dm.distmesh2d(fd, fh, h, (x1, y1, x2, y2),[(x1, y1), (x1, y2), (x2, y1), (x2, y2)])

    plt.show()



    bbox=(x1, y1, x2, y2)

    ii = 0
    b1=[0]
    while ii<p.shape[0]:
     if abs(p[ii,0] - bbox[0]) < 1e-7:
      b1=np.append(b1, [ii])
     ii+=1
    b1=b1[np.arange(1,b1.shape[0])]

    ii = 0
    b2 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 1] - bbox[1]) < 1e-7:
            b2 = np.append(b2, [ii])
        ii += 1
    b2 = b2[np.arange(1, b2.shape[0])]

    ii = 0
    b3 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 0] - bbox[2]) < 1e-7:
            b3 = np.append(b3, [ii])
        ii += 1
    b3 = b3[np.arange(1, b3.shape[0])]

    ii = 0
    b4 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 1] - bbox[3]) < 1e-7:
            b4 = np.append(b4, [ii])
        ii += 1
    b4 = b4[np.arange(1, b4.shape[0])]

    b1+=1
    b2+=1
    b3+=1
    b4+=1

    border_size=max(b1.shape[0],b2.shape[0],b3.shape[0],b4.shape[0])
    b=np.zeros((4,border_size))

    b[0,:]=np.concatenate((b1,np.zeros((1,border_size-b1.shape[0]))),axis=None)
    b[1,:]=np.concatenate((b2,np.zeros((1,border_size-b2.shape[0]))),axis=None)
    b[2,:]=np.concatenate((b3,np.zeros((1,border_size-b3.shape[0]))),axis=None)
    b[3,:]=np.concatenate((b4,np.zeros((1,border_size-b4.shape[0]))),axis=None)

    b=b.astype(int)

    #t+=1
    return p, t, b


def get_plies():
    N=int(input('Number of plies : '))
    PPT = np.zeros((N, 4))
    angles = np.zeros((N, 1))
    thickness = np.zeros((N, 1))
    pho = np.zeros((N,1))


    i=0
    while i<N:
     print('Ply n° {}', i+1)
     theta = float(input('angle : '))
     angles[i] = theta
     EL = float(input('EL : '))
     PPT[i,0] = EL
     ET = float(input('ET : '))
     PPT[i,1] = ET
     vLT = float(input('vLT : '))
     PPT[i,2] = vLT
     GLT = float(input('GLT : '))
     PPT[i,3] = GLT
     pho[i] = float(input('Density : '))
     thickness[i] = float(input('Ply thickness : '))
     i+=1

    TH = sum(thickness) / 2

    return N, PPT, angles, thickness, TH, pho


def constitutive_law(N, PPT, angles, thickness, TH):

    Qprime = np.zeros((3*N, 3))
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    pos = np.zeros((N, 1))
    Q = np.zeros((3*N, 3))

    i=0
    while i<N:
     EL = PPT[i,0]
     ET = PPT[i,1]
     vLT = PPT[i,2]
     GLT = PPT[i,3]
     Q[3 * i, 0] = EL / (1 - (vLT ** 2) * (ET / EL))
     Q[3 * i + 1, 1] = Q[3 * i, 0] * ET / EL
     Q[3 * i, 1] = Q[3 * i + 1, 1] * vLT
     Q[3 * i + 1, 0] = Q[3 * i, 1]
     Q[3 * i + 2, 2] = GLT
     i+=1

    i=0
    while i<N:
     pos[i] = TH - thickness[i]
     TH = pos[i]
     i+=1

    angles = angles*m.pi/180

    PPTxyz = np.zeros((N, 4))
    PPTxyz[:, 0]=(1/ (((np.cos(angles)** 4)/ np.array([PPT[:, 0]]).T)+((np.sin(angles)** 4)/ (np.array([PPT[:, 1]]).T))+((np.sin(angles)** 2)*(np.cos(angles)** 2)*((1/ np.array([PPT[:, 3]]).T) - 2 * (np.array([PPT[:, 2]]).T/ np.array([PPT[:,0]]).T))))).T
    PPTxyz[:, 1]=(1/ (((np.sin(angles)** 4)/ np.array([PPT[:, 0]]).T)+((np.cos(angles)** 4)/ (np.array([PPT[:, 1]]).T))+((np.sin(angles)** 2)*(np.cos(angles)** 2)*((1/ np.array([PPT[:, 3]]).T) - 2 * (np.array([PPT[:, 2]]).T/ np.array([PPT[:,0]]).T))))).T
    PPTxyz[:, 2]=(np.array([PPT[:,0]]).T*((np.array([PPT[:, 2]]).T/ np.array([PPT[:,0]]).T)*(np.sin(angles)** 4 + np.cos(angles)** 4)-(np.sin(angles)** 2)*(np.cos(angles)** 2)*(1/ np.array([PPT[:,0]]).T + 1/ np.array([PPT[:, 1]]).T-1/ np.array([PPT[:, 3]]).T))).T
    PPTxyz[:, 3]=(1/  (2 * (np.sin(angles) ** 2)*(np.cos(angles)** 2)*((2/ np.array([PPT[:,0]]).T) + (2 / np.array([PPT[:, 1]]).T)+(4 * np.array([PPT[:, 2]]).T / np.array([PPT[:,0]]).T)-1 / np.array([PPT[:,3]]).T)+(np.sin(angles)** 4 + np.cos(angles)** 4)/ np.array([PPT[:, 3]]).T)).T


    Qprime[np.arange(0,3*N,3), 0]=np.array(np.array([Q[np.arange(0,3*N,3), 0]]).T*np.cos(angles)** 4 + np.array([Q[np.arange(1,3*N,3), 1]]).T*np.sin(angles)** 4 + 2 * (np.sin(angles)** 2)*(np.cos(angles)** 2)*(2 * np.array([Q[np.arange(2,3*N,3), 2]]).T + np.array([Q[np.arange(0,3*N,3), 1]]).T)).T
    Qprime[np.arange(0,3*N,3), 1]=np.array((np.array([Q[np.arange(0,3*N,3), 0]]).T + np.array([Q[np.arange(1,3*N,3), 1]]).T-4 * np.array([Q[np.arange(2,3*N,3), 2]]).T)*(np.sin(angles)** 2)*(np.cos(angles)** 2)+np.array([Q[np.arange(0,3*N,3), 1]]).T*(np.cos(angles)** 4 + np.sin(angles)** 4)).T
    Qprime[np.arange(1,3*N,3), 0]=Qprime[np.arange(0,3*N,3), 1]
    Qprime[np.arange(0,3*N,3), 2]=np.array((np.array([Q[np.arange(0,3*N,3), 0]]).T - np.array([Q[np.arange(0,3*N,3), 1]]).T-2 * np.array([Q[np.arange(2,3*N,3), 2]]).T)*np.sin(angles)*(np.cos(angles)** 3)+(np.array([Q[np.arange(0,3*N,3), 1]]).T - np.array([Q[np.arange(1,3*N,3), 1]]).T+2 * np.array([Q[np.arange(2,3*N,3), 2]]).T)*np.cos(angles)*(np.sin(angles)** 3)).T
    Qprime[np.arange(2,3*N,3), 0]=Qprime[np.arange(0,3*N,3), 2]
    Qprime[np.arange(1,3*N,3), 1]=np.array(np.array([Q[np.arange(0,3*N,3), 0]]).T*np.sin(angles)** 4 + np.array([Q[np.arange(1,3*N,3), 1]]).T*np.cos(angles)** 4 + 2 * (np.sin(angles)** 2)*(np.cos(angles)** 2)*(2 * np.array([Q[np.arange(2,3*N,3), 2]]).T + np.array([Q[np.arange(0,3*N,3), 1]]).T)).T
    Qprime[np.arange(1,3*N,3), 2]=np.array((np.array([Q[np.arange(0,3*N,3), 0]]).T - np.array([Q[np.arange(0,3*N,3), 1]]).T-2 * np.array([Q[np.arange(2,3*N,3), 2]]).T)*np.cos(angles)*(np.sin(angles)** 3)+(np.array([Q[np.arange(0,3*N,3), 1]]).T - np.array([Q[np.arange(1,3*N,3), 1]]).T+2 * np.array([Q[np.arange(2,3*N,3), 2]]).T)*np.sin(angles)*(np.cos(angles)** 3)).T
    Qprime[np.arange(2,3*N,3), 1]=Qprime[np.arange(1,3*N,3), 2]
    Qprime[np.arange(2,3*N,3), 2]=np.array((np.array([Q[np.arange(0,3*N,3), 0]]).T + np.array([Q[np.arange(1,3*N,3), 1]]).T-2 * (np.array([Q[np.arange(0,3*N,3), 1]]).T + np.array([Q[np.arange(2,3*N,3), 2]]).T))*(np.sin(angles)** 2)*(np.cos(angles)** 2)+np.array([Q[np.arange(2,3*N,3), 2]]).T*(np.cos(angles)** 4 + np.sin(angles)** 4)).T

    i=0
    while i<N:
     hi1 = pos[i]
     hi2 = pos[i] + thickness[i]
     A = A + thickness[i] * Qprime[np.arange(3*i,3*i+3),:]
     B = B + 0.5 * (hi2 ** 2 - hi1 ** 2) * Qprime[np.arange(3*i,3*i+3),:]
     D = D + (1 / 3) * (hi2 ** 3 - hi1 ** 3) * Qprime[np.arange(3*i,3*i+3),:]
     i+=1

    K=np.array([[A[0, 0], A[0, 1], A[0, 2], -B[0, 0], -B[0, 1], -B[0, 2]],
                [A[1, 0], A[1, 1], A[1, 2], -B[1, 0], -B[1, 1], -B[1, 2]],
                [A[2, 0], A[2, 1], A[2, 2], -B[2, 0], -B[2, 1], -B[2, 2]],
                [-B[0, 0], -B[0, 1], -B[0, 2], D[0, 0], D[0, 1], D[0, 2]],
                [-B[1, 0], -B[1, 1], -B[1, 2], D[1, 0], D[1, 1], D[1, 2]],
                [-B[2, 0], -B[2, 1], -B[2, 2], D[2, 0], D[2, 1], D[2, 2]]])

    return pos, Q, Qprime, A, B, D, K


def get_loads(X):
    f =lambda x,y:0
    g =lambda x,y:0
    h =lambda x,y:0
    pointload = -1
    NODALLOAD = np.array([0, 0, 0])
    XY = np.zeros((1,2))
    load = 1
    NODALLOAD1 = np.array([NODALLOAD])
    pointload1 = ([])

    while load!=0:
        load = float(input('Choose Load type : Pressure 1 / Surface Load 2 / Nodal Load 3 /Exit 0 '))
        if load==1:
            P = input('Value (suivant (O,z)) (Pa) :')
            f = eval("lambda x,y:" + P)
            print('\n')
        if load==2:
            gl = input('Value [Ox] (Pa) :')
            g = eval("lambda x,y:" + gl)
            hl = input('Value [Oy] (Pa) :')
            h = eval("lambda x,y:" + hl)
            print('\n')
        if load==3:
            print('Les coordonnées du point d application : [x y] ')
            for j in np.arange(0, 2):
                XY[0,j] = float(input())
            NODALLOAD[0] = float(input('Load coordinates (N) : [Fx     ] '))
            NODALLOAD[1] = float(input('Load coordinates (N) : [   Fy  ] '))
            NODALLOAD[2] = float(input('Load coordinates (N) : [     Fz] '))
            NODALLOAD1 = np.concatenate((NODALLOAD1,np.array([NODALLOAD])),axis=0)
            distload = 10000
            ii=0
            while ii<X.shape[0]:
                if m.sqrt(((X[ii,0]-XY[0,0])**2)+((X[ii,1]-XY[0,1])**2)) <= distload:
                    distload = m.sqrt(((X[ii,0]-XY[0,0])**2)+((X[ii,1]-XY[0,1])**2))
                    pointload = ii
                ii += 1
            pointload1 = np.append(pointload1,pointload)
            pointload1 = pointload1.astype(int)
            print('\n')
    NODALLOAD1 = NODALLOAD1[1::, :]
    return f,g,h,pointload1,NODALLOAD1


def get_boundaryconditions(analysis_type):

    NX1 = lambda y: 0
    NY1 = lambda y: 0
    NX2 = lambda x: 0
    NY2 = lambda x: 0
    NX3 = lambda y: 0
    NY3 = lambda y: 0
    NX4 = lambda x: 0
    NY4 = lambda x: 0
    MY1 = lambda y: 0
    MXY1 = lambda y: 0
    MX2 = lambda x: 0
    MXY2 = lambda x: 0
    MY3 = lambda y: 0
    MXY3 = lambda y: 0
    MX4 = lambda x: 0
    MXY4 = lambda x: 0


    boundaryconditions = np.array([0,0,0,0])
    ENFRCDS = np.zeros((4, 10))
    BC = 9
    if analysis_type[0,0]==3:
     condition = np.array([0,2,5,6,7,8])
    else:
     condition = np.arange(0,9)

    while not np.any(np.equal(BC, condition)):
     if analysis_type[0,0] == 3:
            BC = int(input('Choose boundary conditions (x=x1) :  Fixed 2 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
            print('\n')
     else:
            BC = int(input('Choose boundary conditions (x=x1) :  Membrane Load 1 / Fixed 2 / Moment 3 / Imposed displacements 4 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
            print('\n')
     if BC == 1:
      boundaryconditions[0] = 1
      Nx1 = input('Value [ Ox  ] (N/m) :')
      NX1 = eval("lambda y:" + Nx1)
      Ny1 = input('Value [ Oy  ] (N/m) :')
      NY1 = eval("lambda y:" + Ny1)
      print('\n')
     if BC == 2:
      boundaryconditions[0] = 2
     if BC == 3:
      boundaryconditions[0] = 3
      My1 = input('Value My (N) (Bending) :')
      MY1 = eval("lambda y:" + My1)
      Mxy1 = input('Value Mxy (torsion) (N) :')
      MXY1 = eval("lambda y:" + Mxy1)
      print('\n')
     if BC == 4:
      boundaryconditions[0] = 4
      print('ddl : [1/0 1/0 1/0 1/0 1/0] ')
      for j in np.arange(5, 10):
       ENFRCDS[0, j]=input()
      print('\n')
      print('The imposed displacements : [x y z thetax thetay] ')
      for j in np.arange(0, 5):
       ENFRCDS[0, j]=input()
      print('\n')
     if np.any(np.equal(BC, np.arange(5,9))) :
       boundaryconditions[0] = BC

    BC = 9
    while not np.any(np.equal(BC, condition)):
     if analysis_type[0,0] == 3:
            BC = int(input('Choose boundary conditions (y=y1) : Fixed 2 / Pinned Support (Oxz Plane) 5 / Roller Support (Oxz Plane) 6 /  Roller Support (Oyz Plane) 7 /  Pinned Support (Oyz Plane) 8 / Exit 0 '))
            print('\n')
     else :
            BC = int(input('Choose boundary conditions (y=y1) : Membrane Load 1 / Fixed 2 / Moment 3 / Imposed displacements 4 / Pinned Support (Oxz Plane) 5 / Roller Support (Oxz Plane) 6 /  Roller Support (Oyz Plane) 7 /  Pinned Support (Oyz Plane) 8 / Exit 0 '))
            print('\n')

     if BC == 1:
      boundaryconditions[1] = 1
      Nx2 = input('Value [ Ox  ] (N/m) :')
      NX2 = eval("lambda x:" + Nx2)
      Ny2 = input('Value [ Oy  ] (N/m) :')
      NY2 = eval("lambda x:" + Ny2)
      print('\n')
     if BC == 2:
      boundaryconditions[1] = 2
     if BC == 3:
      boundaryconditions[1] = 3
      Mx2 = input('Mx value (N) (Bending) :')
      MX2 = eval("lambda x:" + Mx2)
      Mxy2 = input('Mxy value (torsion) (N) :')
      MXY2 = eval("lambda x:" + Mxy2)
      print('\n')
     if BC == 4:
      boundaryconditions[1] = 4
      print('ddl : [1/0 1/0 1/0 1/0 1/0] ')
      for j in np.arange(5, 10):
       ENFRCDS[1, j]=input()
      print('\n')
      print('The imposed displacements : [x y z thetax thetay] ')
      for j in np.arange(0, 5):
       ENFRCDS[1, j]=input()
      print('\n')
     if np.any(np.equal(BC, np.arange(5,9))) :
       boundaryconditions[1] = BC


    BC = 9
    while not np.any(np.equal(BC, condition)):
     if analysis_type[0,0] == 3:
            BC = int(input('Choose boundary conditions (x=x2) : Fixed 2 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
            print('\n')
     else:
            BC = int(input('Choose boundary conditions (x=x2) : Membrane Load 1 / Fixed 2 / Moment 3 / Imposed displacements 4 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
            print('\n')

     if BC == 1:
      boundaryconditions[2] = 1
      Nx3 = input('Value [ Ox  ] (N/m) :')
      NX3 = eval("lambda x:" + Nx3)
      Ny3 = input('Value [ Oy  ] (N/m) :')
      NY3 = eval("lambda x:" + Ny3)
      print('\n')
     if BC == 2:
      boundaryconditions[2] = 2
     if BC == 3:
      boundaryconditions[2] = 3
      My3 = input('Value My (N) (Bending) :')
      MY3 = eval("lambda x:" + My3)
      Mxy3 = input('Value Mxy (torsion) (N) :')
      MXY3 = eval("lambda x:" + Mxy3)
      print('\n')
     if BC == 4:
      boundaryconditions[2] = 4
      print('ddl : [1/0 1/0 1/0 1/0 1/0] ')
      for j in np.arange(5, 10):
       ENFRCDS[2, j]=input()
      print('\n')
      print('The imposed displacements : [x y z thetax thetay] ')
      for j in np.arange(0, 5):
       ENFRCDS[2, j]=input()
      print('\n')
     if np.any(np.equal(BC, np.arange(5,9))) :
       boundaryconditions[2] = BC


    BC = 9
    while not np.any(np.equal(BC, condition)):
     if analysis_type[0,0] == 3:
         BC = int(input('Choose boundary conditions (y=y2) : Fixed 2 / Pinned Support (Oxz Plane) 5 / Roller Support (Oxz Plane) 6 /  Roller Support (Oyz Plane) 7 /  Pinned Support (Oyz Plane) 8 / Exit 0 '))
         print('\n')
     else:
         BC = int(input('Choose boundary conditions (y=y2) : Membrane Load 1 / Fixed 2 / Moment 3 / Imposed displacements 4 / Pinned Support (Oxz Plane) 5 / Roller Support (Oxz Plane) 6 /  Roller Support (Oyz Plane) 7 /  Pinned Support (Oyz Plane) 8 / Exit 0 '))
         print('\n')

     if BC == 1:
      boundaryconditions[3] = 1
      Nx4 = input('Value [ Ox  ] (N/m) :')
      NX4 = eval("lambda x:" + Nx4)
      Ny4 = input('Value [ Oy  ] (N/m) :')
      NY4 = eval("lambda x:" + Ny4)
      print('\n')
     if BC == 2:
      boundaryconditions[3] = 2
     if BC == 3:
      boundaryconditions[3] = 3
      Mx4 = input('Mx Value (N) (Bending) :')
      MX4 = eval("lambda x:" + Mx4)
      Mxy4 = input('Mxy Value (torsion) (N) :')
      MXY4 = eval("lambda x:" + Mxy4)
      print('\n')
     if BC == 4:
      boundaryconditions[3] = 4
      print('ddl : [1/0 1/0 1/0 1/0 1/0] ')
      for j in np.arange(5, 10):
       ENFRCDS[3, j]=input()
      print('\n')
      print('Imposed displacements : [x y z thetax thetay] ')
      for j in np.arange(0, 5):
       ENFRCDS[3, j]=input()
      print('\n')
     if np.any(np.equal(BC, np.arange(5,9))) :
       boundaryconditions[3] = BC


    ENFRCDS[:, np.arange(0,5)]=ENFRCDS[:, [0, 1, 2, 4, 3]]
    ENFRCDS[:, 3]=-ENFRCDS[:, 3]
    ENFRCDS[:, np.arange(2, 5)] = -ENFRCDS[:, np.arange(2, 5)]

    ENFRCDS[:, np.arange(5, 10)] = ENFRCDS[:, [5, 6, 7, 9, 8]]

    return NX1,NY1,NX2,NY2,NX3,NY3,NX4,NY4,MY1,MXY1,MX2,MXY2,MY3,MXY3,MX4,MXY4,boundaryconditions,ENFRCDS


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


def show_deformation(p,t,u, v, w, thetax, thetay):
    yes = int(input('Deformation ? Yes 1 / No 0 '))
    if yes == 1:
        component = int(input('exx 1 / eyy 2 / exy 3 ? '))
        th = int(input('Mid Plane 0 / at a thickness z (Oz) 1 ? '))
        if component == 1:
            titledef = 'Strain exx (Ox)'
        elif component == 2:
            titledef = 'Strain eyy (Oy)'
        else:
            titledef = 'Shear Gamma xy (Oxy)'

        if th == 0:
            STRAIN = strain_calc(p, t, u, v)
            nexx = int(STRAIN.size / 3)
            start = (component - 1) * nexx
            finish = component * nexx
            exx = STRAIN[start:finish]
            exx = exx.T
            exx = exx[0]
            exx = np.around(exx, decimals=10)
            fig6 = plt.figure()
            plt.gca().set_aspect('equal')
            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=exx, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig6.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')

        else:
            th = float(input('Thickness = '))
            thSTRAINxx, thSTRAINyy, thSTRAINxy = strain_calc_thick(p, t, u, v, w, thetax, thetay, th)
            fig7 = plt.figure()
            plt.gca().set_aspect('equal')
            if component == 1:
                levels = np.linspace(np.min(thSTRAINxx),np.max(thSTRAINxx),100)
                i=0
                while i<t.shape[0]:
                    exx = thSTRAINxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(thSTRAINyy),np.max(thSTRAINyy),100)
                i=0
                while i<t.shape[0]:
                    eyy = thSTRAINyy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(thSTRAINxy),np.max(thSTRAINxy),100)
                i=0
                while i<t.shape[0]:
                    exy = thSTRAINxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")

            plt.colorbar()
            plt.title(titledef)
            fig7.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')


def show_stress(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness):
    yestr = int(input('Stress ? Yes 1 / No 0 '))
    if yestr == 1:
        component = int(input('Sxx 1 / Syy 2 / Sxy 3 ? '))
        th = float(input('Mid Plane 0 / at a thickness z (Oz) =  ? '))
        if component == 1:
            titledef = 'Stress Sxx (Ox)'
        elif component == 2:
            titledef = 'Stress Syy (Oy)'
        else:
            titledef = 'Stress xy (Oxy)'

        if th == 0:
            sxx, syy, sxy = stress_calc(p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness)

            fig8 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx, edgecolors='k')
            elif component == 2:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=syy, edgecolors='k')
            else:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxy, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig8.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')

        else:
            sxx, syy, sxy = stress_calc(p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness)
            fig9 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                levels = np.linspace(np.min(sxx),np.max(sxx),100)
                i=0
                while i<t.shape[0]:
                    exx = sxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")


            plt.colorbar()
            plt.title(titledef)
            fig9.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')


def show_displacement_thickness(p,u, v, thetax, thetay, pos, thickness, N):
    yes2 = int(input('plot Displacement/thickness at a point ? Yes 1 / No 0 '))
    if yes2 == 1:
        x = float(input('x coordinate of the point  ? '))
        y = float(input('y coordinate of the point  ? '))
        component = int(input('u (Ox) 1 / v(Oy) 2  ? '))

        yplot = [pos[N - 1], pos[0] + thickness[0]]

        pointu = 0
        distu = 10000

        i = 0
        while i < p.shape[0]:
            if m.sqrt(((p[i, 0] - x) ** 2) + ((p[i, 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[i, 0] - x) ** 2) + ((p[i, 1] - y) ** 2))
                pointu = i
            i += 1

        fig10 = plt.figure()
        if component == 1:
            xplot = [u[pointu] - pos[N - 1] * thetay[pointu], u[pointu] - (pos[0] + thickness[0]) * thetay[pointu]]
            titledef = 'Displacement u (Ox) by lmainate thickness '
            labeldef = 'u'
        elif component == 2:
            xplot = [v[pointu] - pos[N - 1] * thetax[pointu], v[pointu] - (pos[0] + thickness[0]) * thetax[pointu]]
            titledef = 'Displacement v (Oy) by lmainate thickness '
            labeldef = 'v'

        plt.plot(xplot, yplot, 'b-', label=labeldef)

        i = 0
        while i < N:
            plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
            i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig10.savefig(results_dir + titledef + ' .png')
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
        plt.show()
        print('press any key to continue')


def show_strain_thickness(p,t,u, v, w, thetax, thetay, pos, thickness, N):
    yes3 = int(input('plot Strain/thickness at an element ? Yes 1 / No 0 '))
    if yes3 == 1:

        thSTRAINxx1, thSTRAINyy1, thSTRAINxy1 = strain_calc_thick(p, t, u, v, w, thetax, thetay, pos[N - 1])
        thSTRAINxx2, thSTRAINyy2, thSTRAINxy2 = strain_calc_thick(p, t, u, v, w, thetax, thetay, pos[0] + thickness[0])
        x = float(input('x coordinate of a close point  ? '))
        y = float(input('y coordinate of a close point  ? '))
        component = int(input('exx (Ox) 1 / eyy(Oy) 2 / Gammaxy 3 (Oxy) ? '))

        yplot = [pos[N - 1], pos[0] + thickness[0]]

        pointu = 0
        distu = 10000

        i = 0
        while i < t.shape[0]:
            if x <= np.max(p[t[i,:],0]) and  np.min(p[t[i,:],0]) <= x and  np.min(p[t[i,:],1]) <= y and y <= np.max(p[t[i,:],1]):
                elementu = i
                break
            i += 1


        i = 0
        while i < 3:
            if m.sqrt(((p[t[elementu, i],0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[t[elementu, i], 0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2))
                pointu = i
            i += 1

        fig11 = plt.figure()
        if component == 1:
            xplot = [thSTRAINxx1[elementu,pointu], thSTRAINxx2[elementu,pointu]]
            titledef = 'Strain exx (Ox) by lmainate thickness '
            labeldef = 'exx'
        elif component == 2:
            xplot = [thSTRAINyy1[elementu,pointu], thSTRAINyy2[elementu,pointu]]
            titledef = 'Strain eyy (Oy) by lmainate thickness '
            labeldef = 'eyy'
        else:
            xplot = [thSTRAINxy1[elementu,pointu], thSTRAINxy2[elementu,pointu]]
            titledef = 'Shear Gammaxy (Oy) by lmainate thickness '
            labeldef = 'Gammaxy'

        plt.plot(xplot, yplot, 'b-', label=labeldef)

        i = 0
        while i < N:
            plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
            i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig11.savefig(results_dir + titledef + ' .png')
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
        plt.show()
        print('press any key to continue')


def show_stress_thickness(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness, N):
    yes4 = int(input('plot Stress/thickness at a point ? Yes 1 / No 0 '))
    if yes4 == 1:

        stressused = np.zeros((2 * N, 1))
        yplot = np.zeros((2 * N, 1))

        x = float(input('x coordinate of the point  ? '))
        y = float(input('y coordinate of the point  ? '))
        component = int(input('Sxx (Ox) 1 / Syy(Oy) 2 / Sxy 3 (Oxy) ? '))



        pointu = 0
        distu = 10000

        i = 0
        while i < t.shape[0]:
            if x <= np.max(p[t[i,:],0]) and  np.min(p[t[i,:],0]) <= x and  np.min(p[t[i,:],1]) <= y and y <= np.max(p[t[i,:],1]):
                elementu = i
                break
            i += 1


        i = 0
        while i < 3:
            if m.sqrt(((p[t[elementu, i],0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[t[elementu, i], 0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2))
                pointu = i
            i += 1


        fig11 = plt.figure()
        if component == 1:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)

                if pos[i]==0:

                    stressused = [sxx1[elementu], sxx2[elementu, pointu]]

                elif (pos[i]+thickness[i]) == 0:

                    stressused = [sxx1[elementu, pointu], sxx2[elementu]]

                else:

                    stressused = [sxx1[elementu, pointu], sxx2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Sxx (Ox) by lmainate thickness '
                plt.plot(xplot, yplot, 'b-')
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1

        elif component == 2:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)
                if pos[i] == 0:

                    stressused = [syy1[elementu], syy2[elementu, pointu]]

                elif (pos[i] + thickness[i]) == 0:

                    stressused = [syy1[elementu, pointu], syy2[elementu]]

                else:

                    stressused = [syy1[elementu, pointu], syy2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Syy (Oy) by lmainate thickness '
                plt.plot(xplot, yplot, 'b-')
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1


        else:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)
                if pos[i] == 0:

                    stressused = [sxy1[elementu], sxy2[elementu, pointu]]

                elif (pos[i] + thickness[i]) == 0:

                    stressused = [sxy1[elementu, pointu], sxy2[elementu]]

                else:

                    stressused = [sxy1[elementu, pointu], sxy2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Sxy (Ox) by lmainate thickness '
                labeldef = 'Sxy'
                plt.plot(xplot, yplot, 'b-', label=labeldef)
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig11.savefig(results_dir + titledef + ' .png')
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
        plt.show()
        print('press any key to continue')


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


def show_deformation_LT(p,t,u, v, w, thetax, thetay,pos,thickness, angles):
    nT=t.shape[0]
    yes = int(input('Deformation in principal ply axes ? Yes 1 / No 0 '))
    if yes == 1:
        component = int(input('eLL 1 / eTT 2 / eLT 3 ? '))
        th = int(input('Mid Plane 0 / at a thickness z (Oz) 1 ? '))
        if component == 1:
            titledef = 'Strain eLL (Ox)'
        elif component == 2:
            titledef = 'Strain eTT (Oy)'
        else:
            titledef = 'Shear Gamma LT (Oxy)'

        if th == 0:
            strain_L, strain_T, strain_LT = strain_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,thickness, angles)
            strain_L, strain_T, strain_LT = np.reshape(strain_L,(nT,1)), np.reshape(strain_T,(nT,1)) , np.reshape(strain_LT,(nT,1))
            STRAIN = np.concatenate((strain_L, strain_T, strain_LT),axis=0)
            nexx = int(STRAIN.size / 3)
            start = (component - 1) * nexx
            finish = component * nexx
            exx = STRAIN[start:finish]
            exx = exx.T
            exx = exx[0]
            exx = np.around(exx, decimals=10)
            fig12 = plt.figure()
            plt.gca().set_aspect('equal')
            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=exx, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig12.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')

        else:
            th = float(input('Thickness = '))
            strain_L, strain_T, strain_LT = strain_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,thickness, angles)
            fig13 = plt.figure()

            if component == 1:
                levels = np.linspace(np.min(strain_L),np.max(strain_L),100)
                i=0
                while i<t.shape[0]:
                    exx = strain_L[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(strain_T),np.max(strain_T),100)
                i=0
                while i<t.shape[0]:
                    eyy = strain_T[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(strain_LT),np.max(strain_LT),100)
                i=0
                while i<t.shape[0]:
                    exy = strain_LT[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")

            plt.colorbar()
            plt.title(titledef)
            fig13.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')


def show_stress_LT(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness, angles):
    yestr = int(input('Stress in ply principal axes ? Yes 1 / No 0 '))
    if yestr == 1:
        component = int(input('S LL 1 / S TT 2 / S LT 3 ? '))
        th = float(input('Mid Plane 0 / at a thickness z (Oz) =  ? '))
        if component == 1:
            titledef = 'Stress S LL (Ox)'
        elif component == 2:
            titledef = 'Stress S TT (Oy)'
        else:
            titledef = 'Stress LT (Oxy)'

        if th == 0:
            sxx, syy, sxy = stress_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles)

            fig14 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx, edgecolors='k')
            elif component == 2:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=syy, edgecolors='k')
            else:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxy, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig14.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')

        else:
            sxx, syy, sxy = stress_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles)
            fig15 = plt.figure()

            if component == 1:
                levels = np.linspace(np.min(sxx),np.max(sxx),100)
                i=0
                while i<t.shape[0]:
                    exx = sxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")


            plt.colorbar()
            plt.title(titledef)
            fig15.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')


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


def show_Hoffman_stress(p,t,u,v,w,thetax,thetay,pos,Qprime,thickness, angles):

    yeshoffstr = int(input('Hoffman Stress ? Yes 1 / No 0 '))


    if yeshoffstr == 1:

        th = float(input('thickness = ?'))
        SLLt = float(input('Longitudinal tension strength SLLt = '))
        SLLc = float(input('Longitudinal compression strength SLLc = '))
        STTt = float(input('Transversal tension strength STTt = '))
        STTc = float(input('Transversal compression strength STTc = '))
        SLT = float(input('Shear strength SLT = '))

        titledef = 'Hoffman criterion'

        Hoffman_stress = Hoffman(SLLt, SLLc, STTt, STTc, SLT, p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness, angles)

        if th == 0:

            fig16 = plt.figure()
            plt.gca().set_aspect('equal')

            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=Hoffman_stress, edgecolors='k')

            plt.colorbar()
            plt.title(titledef)
            fig16.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')

        else:
            fig17 = plt.figure()

            levels = np.linspace(np.min(Hoffman_stress), np.max(Hoffman_stress), 100)
            i = 0
            while i < t.shape[0]:
                exx = Hoffman_stress[i, :]
                exx = np.around(exx, decimals=10)
                plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                i += 1

            plt.colorbar()
            plt.title(titledef)
            fig17.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
            plt.show()
            print('press any key to continue')


def ElemMassMat(X, T, ie, gp, Wgauss, pho, thickness):


        pliesnum = pho.size
        pliesthick = np.zeros((pliesnum,1))

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

         L=gp[ii,:]
         N1=L[0]
         N2=L[1]
         N3=L[2]

         P = np.array([[N1],
         [N2],
         [N3],
         [N1 * N2],
         [N2 * N3],
         [N3 * N1],
         [N2 * N1 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,2]) * N1 - (1 + 3 * mu[0,2]) * N2 + N3 * (1 + 3 * mu[0,2]))],
         [N3 * N2 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,0]) * N2 - (1 + 3 * mu[0,0]) * N3 + N1 * (1 + 3 * mu[0,0]))],
         [N1 * N3 ** 2 + 0.5 * N1 * N2 * N3 * (3 * (1 - mu[0,1]) * N3 - (1 + 3 * mu[0,1]) * N1 + N2 * (1 + 3 * mu[0,1]))]])

         P=P[:,0]

         N = np.array([ [N1],
                        [N1],
              [P[0] - P[3] + P[5] + 2 * (P[6] - P[8])],
              [- (-c[0,1] * (P[8] - P[5]) - c[0,2] * P[6])],
              [- b[0,1] * (P[8] - P[5]) - b[0,2] * P[6]],
                        [0],
                        [N2],
                        [N2],
              [P[1] - P[4] + P[3] + 2 * (P[7] - P[8])],
              [- (-c[0,2] * (P[6] - P[3]) - c[0,0] * P[7])],
              [- b[0,2] * (P[6] - P[3]) - b[0,0] * P[7]],
                        [0],
                        [N3],
                        [N3],
              [P[2] - P[5] + P[4] + 2 * (P[8] - P[7])],
              [- (-c[0,0] * (P[7] - P[4]) - c[0,1] * P[8])],
              [- b[0,0] * (P[7] - P[4]) - b[0,1] * P[8]],
                        [0]])

         N=N[:,0]



         B=np.array([[N[0]  ,0     ,0            ,0            ,0            ,0 ,N[6]  ,0     ,0            ,0            ,0             ,0 ,N[12] ,0     ,0            ,0            ,0           ,0],
                     [0     ,N[1]  ,0            ,0            ,0            ,0 ,0     ,N[7]  ,0            ,0            ,0             ,0 ,0     ,N[13] ,0            ,0            ,0           ,0],
                     [0     ,0     ,N[2]        ,N[3]        ,N[4]        ,0 ,0     ,0     ,N[8]        ,N[9]        ,N[10]        ,0 ,0     ,0     ,N[14]       ,N[15]       ,N[16]      ,0]])



         Me += Wgauss[ii] * np.linalg.det(J)*np.dot(B.T, B)
         ii+=1

        elemMass = 0
        jj = 0
        while jj<pliesnum:
            interm = pho[jj] * thickness[jj]
            elemMass += Me * interm[0]
            jj+=1

        return elemMass


def get_type():
    analysis_type = np.zeros((1,2))
    analysis_type[0,0] = int(input('Static 1 / Dynamic 2 / Frequency 3 : '))

    if analysis_type[0,0] == 1:
        analysis_type[0,1] = int(input('Elastic 1 / Plastic 2 :'))

    return analysis_type



def get_transient():
    tf = float(input('Final instant :  [s] '))
    delta_T = float(input('Time step :  [s] '))
    integration = int(input('Time integration : Backward euler scheme 1 / Newmark 2 / Explicit 3 :'))

    transient = {'final_instant':tf, 'time_step':delta_T, 'scheme':integration}

    return transient


def animate_mode(frq,modes,mode_number,modal_indexes,mesh_size,p,t):

    num=500
    freq1 = frq[mode_number-1]
    if freq1!=0:
     t1 = np.linspace(0, 1, num) / freq1
    else:
     t1 = np.linspace(0, 1, num)

    vec = np.zeros((6*mesh_size,1))
    vec[modal_indexes,0] = modes[:,mode_number-1]
    vec = vec[2::6]
    normalise = np.trim_zeros(np.absolute(vec))
    if normalise.size !=0:
        vec = vec*(1/np.max(normalise))

    if freq1!=0:
     y = vec * np.sin(2*m.pi*t1*freq1)
    else:
     y = vec * np.sin(2 * m.pi * t1)


    fig = plt.figure(figsize=(10.2, 6.8), dpi=100)
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, np.max(y))
    ax.set_axis_off()

    def run_anim(i1):
        ax = fig.gca(projection='3d')
        ax.cla()
        ax.set_zlim3d(0, np.max(y))
        ax.set_axis_off()
        plt.title('Animation du mode ' + str(mode_number) + ' de frequence ' + '%.2f' % freq1 + ' Hz')
        running_anim = ax.plot_trisurf(p[:,0], p[:,1], t,y[:,i1], cmap="jet", antialiased=True)

        return running_anim

    ani = FuncAnimation(fig, run_anim, frames=num, interval=500, repeat=True, blit=False)
    #plt.show()

    return ani


def animate_transient(SOL, p, t, delta_T):

    num=500

    fig = plt.figure(figsize=(10.2, 6.8), dpi=100)
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, np.max(SOL))
    ax.set_axis_off()

    def run_anim(i1):
        ax = fig.gca(projection='3d')
        ax.cla()
        ax.set_zlim3d(0, np.max(SOL))
        instant = delta_T*i1
        plt.title('Transient response of the transverse displacement of the plate ' + 't = ' +  '%.2f' % instant + ' s' )
        running_anim = ax.plot_trisurf(p[:,0], p[:,1], t,SOL[2::6,i1], cmap="jet", antialiased=True)

        return running_anim

    ani = FuncAnimation(fig, run_anim, frames=num, interval=500, repeat=False, blit=False)
    #plt.show()

    return ani


def plastic_analysis(X, T, K, Fb, Nincr, limit, angles, thickness, pos, Q, init_sigma):

    TT=T.shape[0]
    Nplies = len(angles)
    i=0
    while i<Nplies:
        if pos[i]*(pos[i]+thickness[i])<=0:
            mid_lay = i
            break
        i +=1

    #The load is divided on equal incremental loads
    deltaFb = Fb/Nincr
    residual = deltaFb
    deltaU = []
    epsxx = []
    epsyy = []
    epsxy = []
    deltaU_step = np.zeros((len(Fb,1)))

    strain = np.zeros((3*TT, 1))
    i=0

    while i<Nincr:

        #residual from internal_forces
        dU = np.linalg.solve(K, residual)
        deltaU_step = deltaU_step + dU

        u = deltaU_step[0::6]
        v = deltaU_step[1::6]
        w = deltaU_step[2::6]
        thetax = deltaU_step[4::6]
        thetay = -deltaU_step[3::6]

        STRAINxx = []
        STRAINyy = []
        STRAINxy = []

        SIGMAXX = []
        SIGMAYY = []
        SIGMAXY = []

        STRAIN = np.zeros((3,1))

        thSTRAINxx = np.zeros((3,1))
        thSTRAINyy = np.zeros((3,1))
        thSTRAINxy = np.zeros((3,1))


        k = 0
        while k<TT:

            strainxx = []
            strainyy = []
            strainxy = []

            sigmaxx = []
            sigmayy = []
            sigmxy = []


            xe = X[T[k, :], :]
            b = np.zeros((1, 3))
            b[0, 0] = xe[1, 1] - xe[2, 1]
            b[0, 1] = xe[2, 1] - xe[0, 1]
            b[0, 2] = xe[0, 1] - xe[1, 1]

            c = np.zeros((1, 3))
            c[0, 0] = xe[2, 0] - xe[1, 0]
            c[0, 1] = xe[0, 0] - xe[2, 0]
            c[0, 2] = xe[1, 0] - xe[0, 0]

            delta = 0.5 * (b[0, 0] * c[0, 1] - b[0, 1] * c[0, 0])
            dN1dx = b[0, 0] / (2 * delta)
            dN1dy = c[0, 0] / (2 * delta)
            dN2dx = b[0, 1] / (2 * delta)
            dN2dy = c[0, 1] / (2 * delta)
            dN3dx = b[0, 2] / (2 * delta)
            dN3dy = c[0, 2] / (2 * delta)

            STRAIN = np.array(
                [[dN1dx * u[T[k, 0]] + dN2dx * u[T[k, 1]] + dN3dx * u[T[k, 2]]],
                 [(dN1dy * v[T[k, 0]] + dN2dy * v[T[k, 1]] + dN3dy * v[T[k, 2]])],
                 [(dN1dy * u[T[k, 0]] + dN2dy * u[T[k, 1]] + dN3dy * u[T[k, 2]] + dN1dx * v[T[k, 0]] + dN2dx * v[
                     T[k, 1]] + dN3dx * v[T[k, 2]])]])

            kappa = k_calc(X,T,w,thetax,thetay,k)

            layer_STRAINxx = []
            layer_STRAINyy = []
            layer_STRAINxy = []

            layer_SIGMAXX = []
            layer_SIGMAYY = []
            layer_SIGMAXY = []

            yieldval = np.zeros((1,Nplies))

            j=0

            while j<Nplies:

                th = pos[j] + thickness[j]/2
                thSTRAINxx[0] = STRAIN[0] + th * kappa[0]
                thSTRAINxx[1] = STRAIN[0] + th * kappa[3]
                thSTRAINxx[2] = STRAIN[0] + th * kappa[6]

                thSTRAINyy[0] = STRAIN[1] + th * kappa[1]
                thSTRAINyy[1] = STRAIN[1] + th * kappa[4]
                thSTRAINyy[2] = STRAIN[1] + th * kappa[7]

                thSTRAINxy[0] = STRAIN[2] + th * kappa[2]
                thSTRAINxy[1] = STRAIN[2] + th * kappa[5]
                thSTRAINxy[2] = STRAIN[2] + th * kappa[8]


                layer_STRAINxx.append(thSTRAINxx)
                layer_STRAINyy.append(thSTRAINyy)
                layer_STRAINxy.append(thSTRAINxy)

                thSTRAIN = np.array([[thSTRAINxx[i, j]], [thSTRAINyy[i, j]], [thSTRAINxy[i, j]]])
                thstress = np.dot(Qprime[3 * ply:3 * ply + 3, :], thSTRAIN)

            # ?
            STRAINxx.append(layer_STRAINxx[chosen_layer])
            STRAINyy.append(layer_STRAINyy[chosen_layer])
            STRAINxy.append(layer_STRAINxy[chosen_layer])

            theta = angles[chosen_layer]

            Tr = np.array([[m.cos(theta)**2,               m.sin(theta)**2,         2*m.sin(theta)*m.cos(theta)],
                           [m.sin(theta)**2,               m.cos(theta)**2,         -2*m.sin(theta)*m.cos(theta)],
                           [-m.sin(theta)*m.cos(theta),   m.sin(theta)*m.cos(theta), (m.cos(theta)**2)-m.sin(theta)**2]])

            # Checking the one that has the biggest yield function

            #get the strain of that layer

        # Tangent_Stiffness from Q from file


            #Finding the new tangent matrix

            #the file Returns also the sigma state to compute
            #the internal forces
        deltaU.append(deltaU_step)
        epsxx.append(STRAINxx)
        epsyy.append(STRAINyy)
        epsxy.append(STRAINxy)

    return 0



def get_plastic():

    Nincr = int(input('Divide the load on how many increments : ? '))
    limit = np.zeros((3,3))
    X = float(input('Longitudinal Yield limit ? X = '))
    Y = float(input('Transverse Yield limit ? Y = '))
    SLT = float(input('Shear Yield limit ? SLT = '))

    limit[0,0] = 1/X
    limit[1,1] = 1/Y
    limit[2,2] = 1/SLT

    return Nincr, limit
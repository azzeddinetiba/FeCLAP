# -*-coding:Latin-1 -*

import numpy as np
import math as m

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

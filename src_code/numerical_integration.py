# -*-coding:Latin-1 -*

import numpy as np
import math as m

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

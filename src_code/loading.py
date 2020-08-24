# -*-coding:Latin-1 -*

from numerical_integration import *
import math as m

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

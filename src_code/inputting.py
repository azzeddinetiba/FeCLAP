# -*-coding:Latin-1 -*

import numpy as np
nborders = 4

def get_plies(analysis_type):
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
     if analysis_type[0,0] != 1:
        pho[i] = float(input('Density : '))
     thickness[i] = float(input('Ply thickness : '))
     i+=1

    TH = sum(thickness) / 2

    return N, PPT, angles, thickness, TH, pho

def get_boundaryconditions(analysis_type):

    """ NX1 = lambda y: 0
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
    MXY4 = lambda x: 0"""

    NX = []
    NY = []
    Mben = []
    Mtor = []

    coor = ["y", "x"]


    boundaryconditions = np.array([0,0,0,0])
    ENFRCDS = np.zeros((nborders, 10))
    BC = 9
    if analysis_type[0,0]==3:
     condition = np.array([0,2,5,6,7,8])
    else:
     condition = np.arange(0,9)

    ii = 0
    while ii < nborders:
        BC = 9
        while not np.any(np.equal(BC, condition)):
         if analysis_type[0,0] == 3:
                BC = int(input('Choose boundary conditions (x=x1) :  Fixed 2 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
                print('\n')
         else:
                BC = int(input('Choose boundary conditions (x=x1) :  Membrane Load 1 / Fixed 2 / Moment 3 / Imposed displacements 4 / Pinned Support (Oyz Plane) 5 / Roller Support (Oyz Plane) 6 /  Roller Support (Oxz Plane) 7 /  Pinned Support (Oxz Plane) 8 / Exit 0 '))
                print('\n')
         if BC == 1:
          boundaryconditions[ii] = 1
          Nx = input('Value [ Ox  ] (N/m) :')
          NX.append(eval("lambda "+coor[ii%2]+":" + Nx))
          Ny = input('Value [ Oy  ] (N/m) :')
          NY.append(eval("lambda "+coor[ii%2]+":" + Ny))
          print('\n')
         if BC == 2:
          boundaryconditions[ii] = 2
         if BC == 3:
          boundaryconditions[ii] = 3
          Mbn = input("Value M"+coor[ii%2]+" (N) (Bending) :")
          Mben.append(eval("lambda "+coor[ii%2]+":" + Mbn))
          Mtr = input('Value Mxy (torsion) (N) :')
          Mtor.append(eval("lambda "+coor[ii%2]+":" + Mtr))
          print('\n')
         if BC == 4:
          boundaryconditions[ii] = 4
          print('ddl : [1/0 1/0 1/0 1/0 1/0] ')
          for j in np.arange(5, 10):
           ENFRCDS[ii, j]=input()
          print('\n')
          print('The imposed displacements : [x y z thetax thetay] ')
          for j in np.arange(0, 5):
           ENFRCDS[ii, j]=input()
          print('\n')
         if np.any(np.equal(BC, np.arange(5,9))):
           boundaryconditions[0] = BC

         ii+=1


    ENFRCDS[:, np.arange(0,5)]=ENFRCDS[:, [0, 1, 2, 4, 3]]
    ENFRCDS[:, 3]=-ENFRCDS[:, 3]
    ENFRCDS[:, np.arange(2, 5)] = -ENFRCDS[:, np.arange(2, 5)]

    ENFRCDS[:, np.arange(5, 10)] = ENFRCDS[:, [5, 6, 7, 9, 8]]

    modes_numb = 0
    if analysis_type[0,0] == 3 and analysis_type[0,2] == 0:
        modes_numb = int(input('Number of modes extracted ? '))

    boundary_load = {'NX':NX, 'NY':NY, 'Mbending':Mben, 'Mtorsion':Mtor,\
                     'boundaryconditions':boundaryconditions, 'ENFRCDS':ENFRCDS,'modes':modes_numb}

    return boundary_load

def get_type():
    analysis_type = np.zeros((1,3))
    analysis_type[0,0] = int(input('Static 1 / Dynamic 2 / Frequency 3 : '))

    if analysis_type[0,0] == 1:
        analysis_type[0,1] = int(input('Elastic 1 / Plastic 2 :'))

    analysis_type[0, 2] = int(input('Solver : Sparse 0 / Dense 1 : '))

    return analysis_type



def get_transient():
    tf = float(input('Final instant :  [s] '))
    delta_T = float(input('Time step :  [s] '))
    integration = int(input('Time integration : Backward euler scheme 1 / Newmark 2 / Explicit 3 :'))

    beta = 0
    gamma = 0
    x = 0
    xdot = 0
    if integration == 2:
        beta = float(input('Beta  :'))
        gamma = float(input('gamma :'))
    if integration == 2 or integration == 3:
        x = float(input('Initial displacement : ? [m]'))
        xdot = float(input('Initial velocity : ? [m/s]'))

    transient = {'final_instant':tf, 'time_step':delta_T, 'scheme':integration,\
                 'Beta': beta, 'Gamma':gamma, 'init_disp':x, 'init_V':xdot}

    return transient


def get_plastic():

    Nincr = int(input('Divide the load on how many increments : ? '))
    Niter = int(input('Maximum Number of iterations : ? '))


    limit = np.zeros((3,3))
    X = float(input('Longitudinal Yield limit ? X = '))
    Y = float(input('Transverse Yield limit ? Y = '))
    SLT = float(input('Shear Yield limit ? SLT = '))

    limit[0,0] = 1/X
    limit[1,1] = 1/Y
    limit[2,2] = 1/SLT


    plast_param = {'increments':Nincr, 'iterations':Niter, 'yield_limit':limit}

    return plast_param

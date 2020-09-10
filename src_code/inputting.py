# -*-coding:Latin-1 -*

import numpy as np

def get_plies(analysis_type):
    N=int(input('Number of plies : '))
    PPT = np.zeros((N, 4))
    angles = np.zeros((N, 1))
    thickness = np.zeros((N, 1))
    pho = np.zeros((N,1))


    i=0
    while i<N:
     print('Ply n� {}', i+1)
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

    modes_numb = 0
    if analysis_type[0,0] == 3 and analysis_type[0,2] == 0:
        modes_numb = int(input('Number of modes extracted ? '))

    boundary_load = {'NX1':NX1, 'NY1':NY1, 'NX2':NX2, 'NY2':  NY2, 'NX3': NX3,\
                     'NY3':NY3,'NX4':NX4, 'NY4':NY4, 'MY1':MY1, 'MXY1':MXY1,\
                     'MX2':MX2, 'MXY2':MXY2, 'MY3':MY3, 'MXY3':MXY3, 'MX4':MX4,\
                     'MXY4':MXY4,'boundaryconditions':boundaryconditions, 'ENFRCDS':ENFRCDS,'modes':modes_numb}

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

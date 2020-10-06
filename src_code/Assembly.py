# -*-coding:Latin-1 -*

from src_code.core import *
from src_code.loading import *
from src_code.boundary_conditions import *
from scipy import sparse as sp

def Assembly2D(X,T,surface_load,Wgauss,gp,Ngauss,Klaw,pho,thickness,analysis_type):

    Nn=6*X.shape[0]
    Nt=T.shape[0]

    if analysis_type[0, 2] == 1:

        K=np.zeros((Nn,Nn))
        F=np.zeros((Nn,1))
        M=np.zeros((Nn,Nn))

    else:

        K = sp.lil_matrix((Nn,Nn))
        F = sp.lil_matrix((Nn,1))
        M = sp.lil_matrix((Nn,Nn))

    ie=0
    while ie < Nt :
     Tie=T[ie,:]+1
     Tie1=np.concatenate((np.arange(6*Tie[0]-5,6*Tie[0]+1),np.arange(6*Tie[1]-5,6*Tie[1]+1),np.arange(6*Tie[2]-5,6*Tie[2]+1)),0)
     if analysis_type[0,2] == 1:
         Ke = ElemMat(X, T, ie, gp, Wgauss, Klaw)
         Fe = SMelem(surface_load, X, T, ie, Ngauss, Wgauss)
     else:
         Ke = sp.lil_matrix(ElemMat(X, T, ie, gp, Wgauss, Klaw))
         Fe = sp.lil_matrix(SMelem(surface_load, X, T, ie, Ngauss, Wgauss))
     K[np.ix_(Tie1-1,Tie1-1)]+=Ke
     F[Tie1-1,:]+=Fe
     if analysis_type[0,0] !=1:
         if analysis_type[0, 2] == 1:
             Me = ElemMassMat(X, T, ie, gp, Wgauss, pho, thickness)
         else:
             Me = sp.lil_matrix(ElemMassMat(X, T, ie, gp, Wgauss, pho, thickness))

         M[np.ix_(Tie1 - 1, Tie1 - 1)] += Me

     ie+=1

    return K,M,F


def applying_BC(total_loading,X,T,b,box,K,F, analysis_type,*args):

    isthereM = len(args)
    if isthereM != 0:
        M = args[0]
    else:
        M=[]

    boundary_load = total_loading['Bc']
    surface_nodal_load = total_loading['surf_node']
    nodal_load = surface_nodal_load['node']

    NX1 = boundary_load['NX1']
    NY1 = boundary_load['NY1']
    NX2 = boundary_load['NX2']
    NY2 = boundary_load['NY2']
    NX3 = boundary_load['NX3']
    NY3 = boundary_load['NY3']
    NX4 = boundary_load['NX4']
    NY4 = boundary_load['NY4']
    MY1 = boundary_load['MY1']
    MXY1 = boundary_load['MXY1']
    MX2 = boundary_load['MX2']
    MXY2 = boundary_load['MXY2']
    MY3 = boundary_load['MY3']
    MXY3 = boundary_load['MXY3']
    MX4 = boundary_load['MX4']
    MXY4 = boundary_load['MXY4']
    boundaryconditions = boundary_load['boundaryconditions']
    ENFRCDS = boundary_load['ENFRCDS']


    pointload = nodal_load['coord']
    NODALLOAD = nodal_load['value']


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

      if isthereM != 0:

          M[border1-1,:]=0
          M[:, border1-1]=0
          M[np.ix_(border1 - 1, border1 - 1)] = np.eye(border1.size)

      F[border1-1] = 0
      K[np.ix_(border1-1, border1-1)] = np.eye(border1.size)


    elif boundaryconditions[0] == 3:
      F[border1[np.arange(3,border1.size,6)]-1,:] += BCMAssembly(X, T, b1, MY1, 2, x1, 1)
      F[border1[np.arange(4,border1.size,6)]-1,:] += BCMAssembly(X, T, b1, MXY1, 2, x1, 2)
    elif boundaryconditions[0] == 4:
      srch=np.nonzero(ENFRCDS2[0,:])
      for c in srch[0]:
       K[border1[np.arange(c,border1.size,6)]-1,:]=0
       if isthereM != 0:
           M[border1[np.arange(c,border1.size,6)]-1,:]=0
           M[np.ix_(border1[np.arange(c, border1.size, 6)] - 1, border1[np.arange(c, border1.size, 6)] - 1)] -= diag_mat(M[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)],analysis_type[0,2]) - np.eye(border1[np.arange(c,border1.size,6)].size)

       F[border1[np.arange(c,border1.size,6)]-1]=ENFRCDS1[0, c]
       #K[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)]=np.eye(border1[np.arange(c,border1.size,6)].size)
       K[np.ix_(border1[np.arange(c, border1.size, 6)] - 1, border1[np.arange(c, border1.size, 6)] - 1)] -= diag_mat(K[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)],analysis_type[0,2]) - np.eye(border1[np.arange(c,border1.size,6)].size)

    elif boundaryconditions[0] == 5:
        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        if isthereM != 0:

            M[border1[::6] - 1, :] = 0
            M[:, border1[::6] - 1] = 0
            M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
            M[border1[1::6] - 1, :] = 0
            M[:, border1[1::6] - 1] = 0
            M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
            M[border1[2::6] - 1, :] = 0
            M[:, border1[2::6] - 1] = 0
            M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
            M[border1[3::6] - 1, :] = 0
            M[:, border1[3::6] - 1] = 0
            M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        K[border1[3::6] - 1, :] = 0
        K[:, border1[3::6] - 1] = 0
        F[border1[3::6] - 1] = 0
        K[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

    elif boundaryconditions[0] == 6:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0
        if isthereM != 0:

            M[border1[1::6] - 1, :] = 0
            M[:, border1[1::6] - 1] = 0
            M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
            M[border1[2::6] - 1, :] = 0
            M[:, border1[2::6] - 1] = 0
            M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
            M[border1[3::6] - 1, :] = 0
            M[:, border1[3::6] - 1] = 0
            M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        K[border1[3::6] - 1, :] = 0
        K[:, border1[3::6] - 1] = 0
        F[border1[3::6] - 1] = 0
        K[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

    elif boundaryconditions[0] == 7:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0

        if isthereM != 0:

            M[border1[1::6] - 1, :] = 0
            M[:, border1[1::6] - 1] = 0
            M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
            M[border1[::6] - 1, :] = 0
            M[:, border1[::6] - 1] = 0
            M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
            M[border1[4::6] - 1, :] = 0
            M[:, border1[4::6] - 1] = 0
            M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        K[border1[4::6] - 1, :] = 0
        K[:, border1[4::6] - 1] = 0
        F[border1[4::6] - 1] = 0
        K[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

    elif boundaryconditions[0] == 8:
        K[border1[1::6] - 1, :] = 0
        K[:, border1[1::6] - 1] = 0

        if isthereM != 0:

            M[border1[1::6] - 1, :] = 0
            M[:, border1[1::6] - 1] = 0
            M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
            M[border1[::6] - 1, :] = 0
            M[:, border1[::6] - 1] = 0
            M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
            M[border1[2::6] - 1, :] = 0
            M[:, border1[2::6] - 1] = 0
            M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
            M[border1[4::6] - 1, :] = 0
            M[:, border1[4::6] - 1] = 0
            M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

        F[border1[1::6] - 1] = 0
        K[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
        K[border1[::6] - 1, :] = 0
        K[:, border1[::6] - 1] = 0
        F[border1[::6] - 1] = 0
        K[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
        K[border1[2::6] - 1, :] = 0
        K[:, border1[2::6] - 1] = 0
        F[border1[2::6] - 1] = 0
        K[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
        K[border1[4::6] - 1, :] = 0
        K[:, border1[4::6] - 1] = 0
        F[border1[4::6] - 1] = 0
        K[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)





    if boundaryconditions[1] == 1:
      F[border2[np.arange(0,border2.size,6)]-1,:] += BCAssembly(X, T, b2, NX2, 1, y1)
      F[border2[np.arange(1,border2.size,6)]-1,:] += BCAssembly(X, T, b2, NY2, 1, y1)
      if boundaryconditions[0] == 2:
       K[border2[np.arange(0,6)]-1,:]=0
       K[:, border2[np.arange(0,6)]-1]=0

       if isthereM != 0:

           M[border2[np.arange(0,6)]-1,:]=0
           M[:, border2[np.arange(0,6)]-1]=0
           M[np.ix_(border2[np.arange(0, 6)] - 1, border2[np.arange(0, 6)] - 1)] = np.eye(6)

       F[border2[np.arange(0,6)]-1] = 0
       K[np.ix_(border2[np.arange(0,6)]-1, border2[np.arange(0,6)]-1)] = np.eye(6)

    elif boundaryconditions[1] == 2:
      K[border2-1,:] = 0
      K[:, border2-1] = 0

      if isthereM != 0:

          M[border2-1,:] = 0
          M[:, border2-1] = 0
          M[np.ix_(border2 - 1, border2 - 1)] = np.eye(border2.size)

      F[border2-1] = 0
      K[np.ix_(border2-1, border2-1)] = np.eye(border2.size)


    elif boundaryconditions[1] == 3:
     F[border2[np.arange(4,border2.size,6)]-1]+= BCMAssembly(X, T, b2, MX2, 1, y1, 2)
     F[border2[np.arange(3,border2.size,6)]-1]+= BCMAssembly(X, T, b2, MXY2, 1, y1, 1)
     if boundaryconditions[0] == 2:
       K[border1-1,:]=0
       K[:, border1-1]=0
       F[border1-1] = 0
       K[np.ix_(border1-1, border1-1)] = np.eye(border1.size)

       if isthereM != 0:

           M[border1-1,:]=0
           M[:, border1-1]=0
           M[np.ix_(border1-1, border1-1)] = np.eye(border1.size)
    elif boundaryconditions[1] == 4:
      srch=np.nonzero(ENFRCDS2[1,:])
      for c in srch[0]:
       K[border2[np.arange(c,border2.size,6)]-1,:]=0
       F[border2[np.arange(c,border2.size,6)]-1,:]=ENFRCDS1[1, c]
       #K[np.ix_(border2[np.arange(c,border2.size,6)]-1, border2[np.arange(c,border2.size,6)]-1)]=np.eye(border2[np.arange(c,border2.size,6)].size)
       K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)] -= diag_mat(K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)],analysis_type[0,2])  - np.eye(border2[np.arange(c,border2.size,6)].size)

       if isthereM != 0:

           M[border2[np.arange(c,border2.size,6)]-1,:]=0
           M[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)] -= diag_mat(K[np.ix_(border2[np.arange(c, border2.size, 6)] - 1, border2[np.arange(c, border2.size, 6)] - 1)],analysis_type[0,2])  - np.eye(border2[np.arange(c,border2.size,6)].size)


    elif boundaryconditions[1] == 5:
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0

        if isthereM != 0:

            M[border2[::6] - 1, :] = 0
            M[:, border2[::6] - 1] = 0
            M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
            M[border2[1::6] - 1, :] = 0
            M[:, border2[1::6] - 1] = 0
            M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
            M[border2[2::6] - 1, :] = 0
            M[:, border2[2::6] - 1] = 0
            M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
            M[border2[4::6] - 1, :] = 0
            M[:, border2[4::6] - 1] = 0
            M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0
        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        K[border2[4::6] - 1, :] = 0
        K[:, border2[4::6] - 1] = 0
        F[border2[4::6] - 1] = 0
        K[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

    elif boundaryconditions[1] == 6:
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        if isthereM != 0:

            M[border2[::6] - 1, :] = 0
            M[:, border2[::6] - 1] = 0
            M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
            M[border2[2::6] - 1, :] = 0
            M[:, border2[2::6] - 1] = 0
            M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
            M[border2[4::6] - 1, :] = 0
            M[:, border2[4::6] - 1] = 0
            M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        K[border2[4::6] - 1, :] = 0
        K[:, border2[4::6] - 1] = 0
        F[border2[4::6] - 1] = 0
        K[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

    elif boundaryconditions[1] == 7:
        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0

        if isthereM != 0:

            M[border2[1::6] - 1, :] = 0
            M[:, border2[1::6] - 1] = 0
            M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
            M[border2[::6] - 1, :] = 0
            M[:, border2[::6] - 1] = 0
            M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
            M[border2[3::6] - 1, :] = 0
            M[:, border2[3::6] - 1] = 0
            M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        K[border2[3::6] - 1, :] = 0
        K[:, border2[3::6] - 1] = 0
        F[border2[3::6] - 1] = 0
        K[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

    elif boundaryconditions[1] == 8:
        K[border2[1::6] - 1, :] = 0
        K[:, border2[1::6] - 1] = 0

        if isthereM != 0:

            M[border2[1::6] - 1, :] = 0
            M[:, border2[1::6] - 1] = 0
            M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
            M[border2[::6] - 1, :] = 0
            M[:, border2[::6] - 1] = 0
            M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
            M[border2[2::6] - 1, :] = 0
            M[:, border2[2::6] - 1] = 0
            M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
            M[border2[3::6] - 1, :] = 0
            M[:, border2[3::6] - 1] = 0
            M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

        F[border2[1::6] - 1] = 0
        K[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
        K[border2[::6] - 1, :] = 0
        K[:, border2[::6] - 1] = 0
        F[border2[::6] - 1] = 0
        K[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
        K[border2[2::6] - 1, :] = 0
        K[:, border2[2::6] - 1] = 0
        F[border2[2::6] - 1] = 0
        K[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
        K[border2[3::6] - 1, :] = 0
        K[:, border2[3::6] - 1] = 0
        F[border2[3::6] - 1] = 0
        K[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)





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

      if isthereM != 0:

        M[border3-1,:]=0
        M[:, border3 - 1] = 0
        M[np.ix_(border3 - 1, border3 - 1)] = np.eye(border3.size)

      K[:, border3-1]=0
      F[border3-1] = 0
      K[np.ix_(border3-1, border3-1)] = np.eye(border3.size)


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
       K[np.ix_(border3[np.arange(c,border3.size,6)]-1, border3[np.arange(c,border3.size,6)]-1)] -= diag_mat(K[np.ix_(border3[np.arange(c,border3.size,6)]-1, border3[np.arange(c,border3.size,6)]-1)],analysis_type[0,2]) - np.eye(border3[np.arange(c,border3.size,6)].size)

    elif boundaryconditions[2] == 5:
        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0

        if isthereM != 0:

            M[border3[::6] - 1, :] = 0
            M[:, border3[::6] - 1] = 0
            M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
            M[border3[1::6] - 1, :] = 0
            M[:, border3[1::6] - 1] = 0
            M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
            M[border3[2::6] - 1, :] = 0
            M[:, border3[2::6] - 1] = 0
            M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
            M[border3[3::6] - 1, :] = 0
            M[:, border3[3::6] - 1] = 0
            M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        K[border3[3::6] - 1, :] = 0
        K[:, border3[3::6] - 1] = 0
        F[border3[3::6] - 1] = 0
        K[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)


    elif boundaryconditions[2] == 6:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0
        if isthereM != 0:

            M[border3[1::6] - 1, :] = 0
            M[:, border3[1::6] - 1] = 0
            M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
            M[border3[2::6] - 1, :] = 0
            M[:, border3[2::6] - 1] = 0
            M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
            M[border3[3::6] - 1, :] = 0
            M[:, border3[3::6] - 1] = 0
            M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        K[border3[3::6] - 1, :] = 0
        K[:, border3[3::6] - 1] = 0
        F[border3[3::6] - 1] = 0
        K[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

    elif boundaryconditions[2] == 7:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0

        if isthereM != 0:

            M[border3[1::6] - 1, :] = 0
            M[:, border3[1::6] - 1] = 0
            M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
            M[border3[::6] - 1, :] = 0
            M[:, border3[::6] - 1] = 0
            M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
            M[border3[4::6] - 1, :] = 0
            M[:, border3[4::6] - 1] = 0
            M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0
        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        K[border3[4::6] - 1, :] = 0
        K[:, border3[4::6] - 1] = 0
        F[border3[4::6] - 1] = 0
        K[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

    elif boundaryconditions[2] == 8:
        K[border3[1::6] - 1, :] = 0
        K[:, border3[1::6] - 1] = 0

        if isthereM != 0:

            M[border3[1::6] - 1, :] = 0
            M[:, border3[1::6] - 1] = 0
            M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
            M[border3[::6] - 1, :] = 0
            M[:, border3[::6] - 1] = 0
            M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
            M[border3[2::6] - 1, :] = 0
            M[:, border3[2::6] - 1] = 0
            M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
            M[border3[4::6] - 1, :] = 0
            M[:, border3[4::6] - 1] = 0
            M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

        F[border3[1::6] - 1] = 0
        K[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
        K[border3[::6] - 1, :] = 0
        K[:, border3[::6] - 1] = 0
        F[border3[::6] - 1] = 0
        K[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
        K[border3[2::6] - 1, :] = 0
        K[:, border3[2::6] - 1] = 0
        F[border3[2::6] - 1] = 0
        K[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
        K[border3[4::6] - 1, :] = 0
        K[:, border3[4::6] - 1] = 0
        F[border3[4::6] - 1] = 0
        K[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)





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

      if isthereM != 0:
          M[border4 - 1, :] = 0
          M[:, border4 - 1] = 0
          M[np.ix_(border4 - 1, border4 - 1)] = np.eye(border4.size)

      K[:, border4-1]=0
      F[border4-1] = 0
      K[np.ix_(border4-1, border4-1)] = np.eye(border4.size)


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
       K[np.ix_(border4[np.arange(c, border4.size, 6)] - 1, border4[np.arange(c, border4.size, 6)] - 1)] -= diag_mat(K[np.ix_(border4[np.arange(c, border4.size, 6)] - 1, border4[np.arange(c, border4.size, 6)] - 1)],analysis_type[0,2]) - np.eye(border4[np.arange(c,border4.size,6)].size)

    elif boundaryconditions[3] == 5:
        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0

        if isthereM != 0:

            M[border4[::6] - 1, :] = 0
            M[:, border4[::6] - 1] = 0
            M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
            M[border4[1::6] - 1, :] = 0
            M[:, border4[1::6] - 1] = 0
            M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
            M[border4[2::6] - 1, :] = 0
            M[:, border4[2::6] - 1] = 0
            M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
            M[border4[4::6] - 1, :] = 0
            M[:, border4[4::6] - 1] = 0
            M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0
        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0
        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[4::6] - 1, :] = 0
        K[:, border4[4::6] - 1] = 0

        F[border4[4::6] - 1] = 0
        K[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

    elif boundaryconditions[3] == 6:
        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0

        if isthereM != 0:

            M[border4[::6] - 1, :] = 0
            M[:, border4[::6] - 1] = 0
            M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
            M[border4[2::6] - 1, :] = 0
            M[:, border4[2::6] - 1] = 0
            M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
            M[border4[4::6] - 1, :] = 0
            M[:, border4[4::6] - 1] = 0
            M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0
        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[4::6] - 1, :] = 0
        K[:, border4[4::6] - 1] = 0
        F[border4[4::6] - 1] = 0
        K[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

    elif boundaryconditions[3] == 7:
        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0

        if isthereM != 0:

            M[border4[1::6] - 1, :] = 0
            M[:, border4[1::6] - 1] = 0
            M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
            M[border4[::6] - 1, :] = 0
            M[:, border4[::6] - 1] = 0
            M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
            M[border4[3::6] - 1, :] = 0
            M[:, border4[3::6] - 1] = 0
            M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0
        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[3::6] - 1, :] = 0
        K[:, border4[3::6] - 1] = 0
        F[border4[3::6] - 1] = 0
        K[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

    elif boundaryconditions[3] == 8:
        K[border4[1::6] - 1, :] = 0
        K[:, border4[1::6] - 1] = 0

        if isthereM != 0:

            M[border4[1::6] - 1, :] = 0
            M[:, border4[1::6] - 1] = 0
            M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
            M[border4[::6] - 1, :] = 0
            M[:, border4[::6] - 1] = 0
            M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
            M[border4[2::6] - 1, :] = 0
            M[:, border4[2::6] - 1] = 0
            M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
            M[border4[3::6] - 1, :] = 0
            M[:, border4[3::6] - 1] = 0
            M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

        F[border4[1::6] - 1] = 0
        K[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)

        K[border4[::6] - 1, :] = 0
        K[:, border4[::6] - 1] = 0

        F[border4[::6] - 1] = 0
        K[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)

        K[border4[2::6] - 1, :] = 0
        K[:, border4[2::6] - 1] = 0

        F[border4[2::6] - 1] = 0
        K[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)

        K[border4[3::6] - 1, :] = 0
        K[:, border4[3::6] - 1] = 0

        F[border4[3::6] - 1] = 0
        K[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)






    K[np.ix_(np.arange(5,K.shape[0],6), np.arange(5,K.shape[0],6))]=np.eye(int(K.shape[0] / 6))
    if isthereM != 0:
        M[np.ix_(np.arange(5,M.shape[0],6), np.arange(5,M.shape[0],6))]=np.eye(int(M.shape[0] / 6))


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

    Ksize=K.shape[0]
    fixed_borders = np.concatenate((fixed_borders, np.arange(5,Ksize,6)+1))
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
            if analysis_type[0,2] == 1:
                F[6 * (pointload[i] + 1) - 6] += NODALLOAD[i, 0]
                F[6 * (pointload[i] + 1) - 5] += NODALLOAD[i, 1]
                F[6 * (pointload[i] + 1) - 4] += NODALLOAD[i, 2]
            else:
                loading_vector = np.zeros((F.shape[0],1))
                loading_vector[6 * (pointload[i] + 1) - 6] += NODALLOAD[i, 0]
                loading_vector[6 * (pointload[i] + 1) - 5] += NODALLOAD[i, 1]
                loading_vector[6 * (pointload[i] + 1) - 4] += NODALLOAD[i, 2]
                F += sp.lil_matrix(loading_vector)
            i+=1

    return fixed_borders,K,M,F



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


def applying_Fix_q(total_loading, X, T, b, box, F, analysis_type, Nincr1, *args):

        Nincr = Nincr1[0]
        curr_incr = Nincr1[1]

        isthereM = len(args)
        if isthereM != 0:
            M = args[0]

        boundary_load = total_loading['Bc']
        surface_nodal_load = total_loading['surf_node']

        NX1 = boundary_load['NX1']
        NY1 = boundary_load['NY1']
        NX2 = boundary_load['NX2']
        NY2 = boundary_load['NY2']
        NX3 = boundary_load['NX3']
        NY3 = boundary_load['NY3']
        NX4 = boundary_load['NX4']
        NY4 = boundary_load['NY4']
        MY1 = boundary_load['MY1']
        MXY1 = boundary_load['MXY1']
        MX2 = boundary_load['MX2']
        MXY2 = boundary_load['MXY2']
        MY3 = boundary_load['MY3']
        MXY3 = boundary_load['MXY3']
        MX4 = boundary_load['MX4']
        MXY4 = boundary_load['MXY4']
        boundaryconditions = boundary_load['boundaryconditions']
        ENFRCDS = boundary_load['ENFRCDS']

        ind = 0
        while ind < 4:
            i = 0
            while i < b.shape[1]:
                if b[ind, i] == 0:
                    if ind == 0:
                        b1 = b[ind, np.arange(0, i)]
                    elif ind == 1:
                        b2 = b[ind, np.arange(0, i)]
                    elif ind == 2:
                        b3 = b[ind, np.arange(0, i)]
                    elif ind == 3:
                        b4 = b[ind, np.arange(0, i)]
                    break
                elif b[ind, i] != 0 and i == b.shape[1] - 1:
                    if ind == 0:
                        b1 = b[ind, np.arange(0, i + 1)]
                    elif ind == 1:
                        b2 = b[ind, np.arange(0, i + 1)]
                    elif ind == 2:
                        b3 = b[ind, np.arange(0, i + 1)]
                    elif ind == 3:
                        b4 = b[ind, np.arange(0, i + 1)]
                    break
                i += 1
            ind += 1

        ii = 0
        while ii < b1.size:
            min = ii
            kk = ii + 1
            while kk < b1.size:
                if X[b1[kk] - 1, 1] < X[b1[min] - 1, 1]:
                    min = kk

                kk += 1
            rempl = b1[min]
            b1[min] = b1[ii]
            b1[ii] = rempl
            ii += 1

        ii = 0
        while ii < b2.size:
            min = ii
            kk = ii + 1
            while kk < b2.size:
                if X[b2[kk] - 1, 0] < X[b2[min] - 1, 0]:
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
                if X[b3[kk] - 1, 1] < X[b3[min] - 1, 1]:
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
                if X[b4[kk] - 1, 0] < X[b4[min] - 1, 0]:
                    min = kk
                kk += 1

            rempl = b4[min]
            b4[min] = b4[ii]
            b4[ii] = rempl
            ii += 1

            ENFRCDS1 = np.array([ENFRCDS[:, np.arange(0, 5)]])
            ENFRCDS2 = np.array([ENFRCDS[:, np.arange(5, 10)]])
            ENFRCDS1 = ENFRCDS1[0]/Nincr
            load_step = ENFRCDS1
            if curr_incr != 0:
                i=0
                while i<curr_incr:
                    ENFRCDS1 += load_step
                    i+=1
            ENFRCDS2 = ENFRCDS2[0]
            x1 = box[0, 0]
            y1 = box[0, 1]
            x2 = box[1, 0]
            y2 = box[1, 1]

        border1 = np.zeros((1, 6 * b1.size))
        ii = 0
        while ii < b1.size:
            border1[0, np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b1[ii] - 5, 6 * b1[ii] + 1)
            ii += 1
        border1 = border1.astype(int)
        border1 = border1[0]

        border2 = np.zeros((1, 6 * b2.size))
        ii = 0
        while ii < b2.size:
            border2[0, np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b2[ii] - 5, 6 * b2[ii] + 1)
            ii += 1
        border2 = border2.astype(int)
        border2 = border2[0]

        border3 = np.zeros((1, 6 * b3.size))
        ii = 0
        while ii < b3.size:
            border3[0, np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b3[ii] - 5, 6 * b3[ii] + 1)
            ii += 1
        border3 = border3.astype(int)
        border3 = border3[0]

        border4 = np.zeros((1, 6 * b4.size))
        ii = 0
        while ii < b4.size:
            border4[0, np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * b4[ii] - 5, 6 * b4[ii] + 1)
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
            F[border1[np.arange(0, border1.size, 6)] - 1] += BCAssembly(X, T, b1, NX1, 2, x1)
            F[border1[np.arange(1, border1.size, 6)] - 1] += BCAssembly(X, T, b1, NY1, 2, x1)
        elif boundaryconditions[0] == 2:

            if isthereM != 0:
                M[border1 - 1, :] = 0
                M[:, border1 - 1] = 0
                M[np.ix_(border1 - 1, border1 - 1)] = np.eye(border1.size)

            F[border1 - 1] = 0


        elif boundaryconditions[0] == 3:
            F[border1[np.arange(3, border1.size, 6)] - 1, :] += BCMAssembly(X, T, b1, MY1, 2, x1, 1)
            F[border1[np.arange(4, border1.size, 6)] - 1, :] += BCMAssembly(X, T, b1, MXY1, 2, x1, 2)
        elif boundaryconditions[0] == 4:
            srch = np.nonzero(ENFRCDS2[0, :])
            for c in srch[0]:
                F[border1[np.arange(c, border1.size, 6)] - 1] = ENFRCDS1[0, c]
                if isthereM != 0:
                    M[border1[np.arange(c, border1.size, 6)] - 1, :] = 0
                    M[np.ix_(border1[np.arange(c, border1.size, 6)] - 1,
                             border1[np.arange(c, border1.size, 6)] - 1)] -= diag_mat(M[np.ix_(
                        border1[np.arange(c, border1.size, 6)] - 1,
                        border1[np.arange(c, border1.size, 6)] - 1)],analysis_type[0,2]) - np.eye(
                        border1[np.arange(c, border1.size, 6)].size)



        elif boundaryconditions[0] == 5:

            if isthereM != 0:
                M[border1[::6] - 1, :] = 0
                M[:, border1[::6] - 1] = 0
                M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
                M[border1[1::6] - 1, :] = 0
                M[:, border1[1::6] - 1] = 0
                M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
                M[border1[2::6] - 1, :] = 0
                M[:, border1[2::6] - 1] = 0
                M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
                M[border1[3::6] - 1, :] = 0
                M[:, border1[3::6] - 1] = 0
                M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

            F[border1[::6] - 1] = 0

            F[border1[1::6] - 1] = 0

            F[border1[2::6] - 1] = 0

            F[border1[3::6] - 1] = 0

        elif boundaryconditions[0] == 6:

            if isthereM != 0:
                M[border1[1::6] - 1, :] = 0
                M[:, border1[1::6] - 1] = 0
                M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
                M[border1[2::6] - 1, :] = 0
                M[:, border1[2::6] - 1] = 0
                M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
                M[border1[3::6] - 1, :] = 0
                M[:, border1[3::6] - 1] = 0
                M[np.ix_(border1[3::6] - 1, border1[3::6] - 1)] = np.eye(border1[3::6].size)

            F[border1[1::6] - 1] = 0

            F[border1[2::6] - 1] = 0

            F[border1[3::6] - 1] = 0

        elif boundaryconditions[0] == 7:


            if isthereM != 0:
                M[border1[1::6] - 1, :] = 0
                M[:, border1[1::6] - 1] = 0
                M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
                M[border1[::6] - 1, :] = 0
                M[:, border1[::6] - 1] = 0
                M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
                M[border1[4::6] - 1, :] = 0
                M[:, border1[4::6] - 1] = 0
                M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

            F[border1[1::6] - 1] = 0

            F[border1[::6] - 1] = 0

            F[border1[4::6] - 1] = 0

        elif boundaryconditions[0] == 8:


            if isthereM != 0:
                M[border1[1::6] - 1, :] = 0
                M[:, border1[1::6] - 1] = 0
                M[np.ix_(border1[1::6] - 1, border1[1::6] - 1)] = np.eye(border1[1::6].size)
                M[border1[::6] - 1, :] = 0
                M[:, border1[::6] - 1] = 0
                M[np.ix_(border1[::6] - 1, border1[::6] - 1)] = np.eye(border1[::6].size)
                M[border1[2::6] - 1, :] = 0
                M[:, border1[2::6] - 1] = 0
                M[np.ix_(border1[2::6] - 1, border1[2::6] - 1)] = np.eye(border1[2::6].size)
                M[border1[4::6] - 1, :] = 0
                M[:, border1[4::6] - 1] = 0
                M[np.ix_(border1[4::6] - 1, border1[4::6] - 1)] = np.eye(border1[4::6].size)

            F[border1[1::6] - 1] = 0

            F[border1[::6] - 1] = 0

            F[border1[2::6] - 1] = 0

            F[border1[4::6] - 1] = 0

        if boundaryconditions[1] == 1:
            F[border2[np.arange(0, border2.size, 6)] - 1, :] += BCAssembly(X, T, b2, NX2, 1, y1)
            F[border2[np.arange(1, border2.size, 6)] - 1, :] += BCAssembly(X, T, b2, NY2, 1, y1)
            if boundaryconditions[0] == 2:


                if isthereM != 0:
                    M[border2[np.arange(0, 6)] - 1, :] = 0
                    M[:, border2[np.arange(0, 6)] - 1] = 0
                    M[np.ix_(border2[np.arange(0, 6)] - 1, border2[np.arange(0, 6)] - 1)] = np.eye(6)

                F[border2[np.arange(0, 6)] - 1] = 0

        elif boundaryconditions[1] == 2:


            if isthereM != 0:
                M[border2 - 1, :] = 0
                M[:, border2 - 1] = 0
                M[np.ix_(border2 - 1, border2 - 1)] = np.eye(border2.size)

            F[border2 - 1] = 0


        elif boundaryconditions[1] == 3:
            F[border2[np.arange(4, border2.size, 6)] - 1] += BCMAssembly(X, T, b2, MX2, 1, y1, 2)
            F[border2[np.arange(3, border2.size, 6)] - 1] += BCMAssembly(X, T, b2, MXY2, 1, y1, 1)
            if boundaryconditions[0] == 2:

                F[border1 - 1] = 0

                if isthereM != 0:
                    M[border1 - 1, :] = 0
                    M[:, border1 - 1] = 0
                    M[np.ix_(border1 - 1, border1 - 1)] = np.eye(border1.size)
        elif boundaryconditions[1] == 4:
            srch = np.nonzero(ENFRCDS2[1, :])
            for c in srch[0]:
                F[border2[np.arange(c, border2.size, 6)] - 1] = ENFRCDS1[1, c]
                if isthereM != 0:
                    M[border2[np.arange(c, border2.size, 6)] - 1, :] = 0
                    M[np.ix_(border2[np.arange(c, border2.size, 6)] - 1,
                             border2[np.arange(c, border2.size, 6)] - 1)] -= diag_mat(M[np.ix_(
                        border2[np.arange(c, border2.size, 6)] - 1,
                        border2[np.arange(c, border2.size, 6)] - 1)],analysis_type[0,2]) - np.eye(
                        border2[np.arange(c, border2.size, 6)].size)


        elif boundaryconditions[1] == 5:


            if isthereM != 0:
                M[border2[::6] - 1, :] = 0
                M[:, border2[::6] - 1] = 0
                M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
                M[border2[1::6] - 1, :] = 0
                M[:, border2[1::6] - 1] = 0
                M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
                M[border2[2::6] - 1, :] = 0
                M[:, border2[2::6] - 1] = 0
                M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
                M[border2[4::6] - 1, :] = 0
                M[:, border2[4::6] - 1] = 0
                M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

            F[border2[::6] - 1] = 0

            F[border2[1::6] - 1] = 0

            F[border2[2::6] - 1] = 0

            F[border2[4::6] - 1] = 0

        elif boundaryconditions[1] == 6:

            if isthereM != 0:
                M[border2[::6] - 1, :] = 0
                M[:, border2[::6] - 1] = 0
                M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
                M[border2[2::6] - 1, :] = 0
                M[:, border2[2::6] - 1] = 0
                M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
                M[border2[4::6] - 1, :] = 0
                M[:, border2[4::6] - 1] = 0
                M[np.ix_(border2[4::6] - 1, border2[4::6] - 1)] = np.eye(border2[4::6].size)

            F[border2[::6] - 1] = 0

            F[border2[2::6] - 1] = 0

            F[border2[4::6] - 1] = 0

        elif boundaryconditions[1] == 7:


            if isthereM != 0:
                M[border2[1::6] - 1, :] = 0
                M[:, border2[1::6] - 1] = 0
                M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
                M[border2[::6] - 1, :] = 0
                M[:, border2[::6] - 1] = 0
                M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
                M[border2[3::6] - 1, :] = 0
                M[:, border2[3::6] - 1] = 0
                M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

            F[border2[1::6] - 1] = 0

            F[border2[::6] - 1] = 0

            F[border2[3::6] - 1] = 0

        elif boundaryconditions[1] == 8:


            if isthereM != 0:
                M[border2[1::6] - 1, :] = 0
                M[:, border2[1::6] - 1] = 0
                M[np.ix_(border2[1::6] - 1, border2[1::6] - 1)] = np.eye(border2[1::6].size)
                M[border2[::6] - 1, :] = 0
                M[:, border2[::6] - 1] = 0
                M[np.ix_(border2[::6] - 1, border2[::6] - 1)] = np.eye(border2[::6].size)
                M[border2[2::6] - 1, :] = 0
                M[:, border2[2::6] - 1] = 0
                M[np.ix_(border2[2::6] - 1, border2[2::6] - 1)] = np.eye(border2[2::6].size)
                M[border2[3::6] - 1, :] = 0
                M[:, border2[3::6] - 1] = 0
                M[np.ix_(border2[3::6] - 1, border2[3::6] - 1)] = np.eye(border2[3::6].size)

            F[border2[1::6] - 1] = 0

            F[border2[::6] - 1] = 0

            F[border2[2::6] - 1] = 0

            F[border2[3::6] - 1] = 0

        if boundaryconditions[2] == 1:
            F[border3[np.arange(0, border3.size, 6)] - 1] += BCAssembly(X, T, b3, NX3, 2, x2)
            F[border3[np.arange(1, border3.size, 6)] - 1] += BCAssembly(X, T, b3, NY3, 2, x2)
            if boundaryconditions[1] == 2:

                F[border3[np.arange(0, 6)] - 1] = 0
        elif boundaryconditions[2] == 2:

            if isthereM != 0:
                M[border3 - 1, :] = 0
                M[:, border3 - 1] = 0
                M[np.ix_(border3 - 1, border3 - 1)] = np.eye(border3.size)

            F[border3 - 1] = 0


        elif boundaryconditions[2] == 3:
            F[border3[np.arange(3, border3.size, 6)] - 1] += BCMAssembly(X, T, b3, MY3, 2, x2, 1)
            F[border3[np.arange(4, border3.size, 6)] - 1] += BCMAssembly(X, T, b3, MXY3, 2, x2, 2)
            if boundaryconditions[1] == 2:
                F[border2 - 1] = 0

        elif boundaryconditions[2] == 4:
            srch = np.nonzero(ENFRCDS2[2, :])
            for c in srch[0]:
                F[border3[np.arange(c, border3.size, 6)] - 1] = ENFRCDS1[2, c]
                if isthereM != 0:
                    M[border3[np.arange(c, border3.size, 6)] - 1, :] = 0
                    M[np.ix_(border3[np.arange(c, border3.size, 6)] - 1,
                             border3[np.arange(c, border3.size, 6)] - 1)] -= diag_mat(M[np.ix_(
                        border3[np.arange(c, border3.size, 6)] - 1,
                        border3[np.arange(c, border3.size, 6)] - 1)],analysis_type[0,2]) - np.eye(
                        border3[np.arange(c, border3.size, 6)].size)


        elif boundaryconditions[2] == 5:


            if isthereM != 0:
                M[border3[::6] - 1, :] = 0
                M[:, border3[::6] - 1] = 0
                M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
                M[border3[1::6] - 1, :] = 0
                M[:, border3[1::6] - 1] = 0
                M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
                M[border3[2::6] - 1, :] = 0
                M[:, border3[2::6] - 1] = 0
                M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
                M[border3[3::6] - 1, :] = 0
                M[:, border3[3::6] - 1] = 0
                M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

            F[border3[::6] - 1] = 0

            F[border3[1::6] - 1] = 0

            F[border3[2::6] - 1] = 0

            F[border3[3::6] - 1] = 0


        elif boundaryconditions[2] == 6:

            if isthereM != 0:
                M[border3[1::6] - 1, :] = 0
                M[:, border3[1::6] - 1] = 0
                M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
                M[border3[2::6] - 1, :] = 0
                M[:, border3[2::6] - 1] = 0
                M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
                M[border3[3::6] - 1, :] = 0
                M[:, border3[3::6] - 1] = 0
                M[np.ix_(border3[3::6] - 1, border3[3::6] - 1)] = np.eye(border3[3::6].size)

            F[border3[1::6] - 1] = 0

            F[border3[2::6] - 1] = 0

            F[border3[3::6] - 1] = 0

        elif boundaryconditions[2] == 7:


            if isthereM != 0:
                M[border3[1::6] - 1, :] = 0
                M[:, border3[1::6] - 1] = 0
                M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
                M[border3[::6] - 1, :] = 0
                M[:, border3[::6] - 1] = 0
                M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
                M[border3[4::6] - 1, :] = 0
                M[:, border3[4::6] - 1] = 0
                M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

            F[border3[1::6] - 1] = 0

            F[border3[::6] - 1] = 0

            F[border3[4::6] - 1] = 0

        elif boundaryconditions[2] == 8:


            if isthereM != 0:
                M[border3[1::6] - 1, :] = 0
                M[:, border3[1::6] - 1] = 0
                M[np.ix_(border3[1::6] - 1, border3[1::6] - 1)] = np.eye(border3[1::6].size)
                M[border3[::6] - 1, :] = 0
                M[:, border3[::6] - 1] = 0
                M[np.ix_(border3[::6] - 1, border3[::6] - 1)] = np.eye(border3[::6].size)
                M[border3[2::6] - 1, :] = 0
                M[:, border3[2::6] - 1] = 0
                M[np.ix_(border3[2::6] - 1, border3[2::6] - 1)] = np.eye(border3[2::6].size)
                M[border3[4::6] - 1, :] = 0
                M[:, border3[4::6] - 1] = 0
                M[np.ix_(border3[4::6] - 1, border3[4::6] - 1)] = np.eye(border3[4::6].size)

            F[border3[1::6] - 1] = 0

            F[border3[::6] - 1] = 0

            F[border3[2::6] - 1] = 0

            F[border3[4::6] - 1] = 0

        if boundaryconditions[3] == 1:
            F[border4[np.arange(0, border4.size, 6)] - 1] += BCAssembly(X, T, b4, NX4, 1, y2)
            F[border4[np.arange(1, border4.size, 6)] - 1] += BCAssembly(X, T, b4, NY4, 1, y2)
            if boundaryconditions[2] == 2:
                sz4 = border4.size

                F[border4[np.arange(sz4 - 1, sz4 - 7, -1)] - 1] = 0

            if boundaryconditions[0] == 2:

                F[border1[np.arange(0, 6)] - 1] = 0
        elif boundaryconditions[3] == 2:

            if isthereM != 0:
                M[border4 - 1, :] = 0
                M[:, border4 - 1] = 0
                M[np.ix_(border4 - 1, border4 - 1)] = np.eye(border4.size)

            F[border4 - 1] = 0


        elif boundaryconditions[3] == 3:
            F[border4[np.arange(4, border4.size, 6)] - 1] += BCMAssembly(X, T, b4, MX4, 1, y2, 2)
            F[border4[np.arange(3, border4.size, 6)] - 1] += BCMAssembly(X, T, b4, MXY4, 1, y2, 1)
            if boundaryconditions[2] == 2:

                F[border3[np.arange(0, 6)] - 1] = 0
            if boundaryconditions[0] == 2:

                F[border1[np.arange(0, 6)] - 1] = 0


        elif boundaryconditions[3] == 4:
            srch = np.nonzero(ENFRCDS2[3, :])
            for c in srch[0]:
                F[border4[np.arange(c, border4.size, 6)] - 1] = ENFRCDS1[3, c]
                if isthereM != 0:
                    M[border4[np.arange(c, border4.size, 6)] - 1, :] = 0
                    M[np.ix_(border4[np.arange(c, border4.size, 6)] - 1,
                             border4[np.arange(c, border4.size, 6)] - 1)] -= diag_mat(M[np.ix_(
                        border4[np.arange(c, border4.size, 6)] - 1,
                        border4[np.arange(c, border4.size, 6)] - 1)],analysis_type[0,2]) - np.eye(
                        border4[np.arange(c, border4.size, 6)].size)


        elif boundaryconditions[3] == 5:


            if isthereM != 0:
                M[border4[::6] - 1, :] = 0
                M[:, border4[::6] - 1] = 0
                M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
                M[border4[1::6] - 1, :] = 0
                M[:, border4[1::6] - 1] = 0
                M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
                M[border4[2::6] - 1, :] = 0
                M[:, border4[2::6] - 1] = 0
                M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
                M[border4[4::6] - 1, :] = 0
                M[:, border4[4::6] - 1] = 0
                M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

            F[border4[::6] - 1] = 0

            F[border4[1::6] - 1] = 0

            F[border4[2::6] - 1] = 0


            F[border4[4::6] - 1] = 0

        elif boundaryconditions[3] == 6:


            if isthereM != 0:
                M[border4[::6] - 1, :] = 0
                M[:, border4[::6] - 1] = 0
                M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
                M[border4[2::6] - 1, :] = 0
                M[:, border4[2::6] - 1] = 0
                M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
                M[border4[4::6] - 1, :] = 0
                M[:, border4[4::6] - 1] = 0
                M[np.ix_(border4[4::6] - 1, border4[4::6] - 1)] = np.eye(border4[4::6].size)

            F[border4[::6] - 1] = 0

            F[border4[2::6] - 1] = 0


            F[border4[4::6] - 1] = 0

        elif boundaryconditions[3] == 7:


            if isthereM != 0:
                M[border4[1::6] - 1, :] = 0
                M[:, border4[1::6] - 1] = 0
                M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
                M[border4[::6] - 1, :] = 0
                M[:, border4[::6] - 1] = 0
                M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
                M[border4[3::6] - 1, :] = 0
                M[:, border4[3::6] - 1] = 0
                M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

            F[border4[1::6] - 1] = 0


            F[border4[::6] - 1] = 0


            F[border4[3::6] - 1] = 0

        elif boundaryconditions[3] == 8:


            if isthereM != 0:
                M[border4[1::6] - 1, :] = 0
                M[:, border4[1::6] - 1] = 0
                M[np.ix_(border4[1::6] - 1, border4[1::6] - 1)] = np.eye(border4[1::6].size)
                M[border4[::6] - 1, :] = 0
                M[:, border4[::6] - 1] = 0
                M[np.ix_(border4[::6] - 1, border4[::6] - 1)] = np.eye(border4[::6].size)
                M[border4[2::6] - 1, :] = 0
                M[:, border4[2::6] - 1] = 0
                M[np.ix_(border4[2::6] - 1, border4[2::6] - 1)] = np.eye(border4[2::6].size)
                M[border4[3::6] - 1, :] = 0
                M[:, border4[3::6] - 1] = 0
                M[np.ix_(border4[3::6] - 1, border4[3::6] - 1)] = np.eye(border4[3::6].size)

            F[border4[1::6] - 1] = 0


            F[border4[::6] - 1] = 0

            F[border4[2::6] - 1] = 0


            F[border4[3::6] - 1] = 0


        return F

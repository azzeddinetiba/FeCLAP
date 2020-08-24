# -*-coding:Latin-1 -*

from boundary_conditions import *
from Assembly import *
from scipy.linalg import eig

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


def plastic_analysis(X, T, K, Fb, Nincr, limit, angles, thickness, pos, Q, init_sigma):

    # ------------      UNDER CONSTRUCTION      ---------------

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

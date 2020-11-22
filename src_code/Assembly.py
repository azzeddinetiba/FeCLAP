# -*-coding:Latin-1 -*

from src_code.core import *
from src_code.loading import *
from src_code.boundary_conditions import *
from scipy import sparse as sp
nborders = 4

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


    NX = boundary_load['NX']
    NY = boundary_load['NY']
    Mben = boundary_load['Mbending']
    Mtor = boundary_load['Mtorsion']
    boundaryconditions = boundary_load['boundaryconditions']
    ENFRCDS = boundary_load['ENFRCDS']

    coor = [2, 1]

    pointload = nodal_load['coord']
    NODALLOAD = nodal_load['value']


    ENFRCDS1 = np.array([ENFRCDS[:, np.arange(0,5)]])
    ENFRCDS2 = np.array([ENFRCDS[:, np.arange(5,10)]])
    ENFRCDS1 = ENFRCDS1[0]
    ENFRCDS2 = ENFRCDS2[0]
    x1 = box[0,0]
    y1 = box[0,1]
    x2 = box[1,0]
    y2 = box[1,1]
    fixed_coor = box.flatten()

    borderss, border, border_size, everyborder = get_boundaries(X, b, nborders)

    i = 0
    count_N = 0
    count_Mom = 0
    while i < nborders:

        if boundaryconditions[i] == 1:
            F[border[i][np.arange(0, border[i].size, 6)] - 1] += BCAssembly(X, T, borderss[i], NX[count_N], coor[i%2], fixed_coor[i])
            F[border[i][np.arange(1, border[i].size, 6)] - 1] += BCAssembly(X, T, borderss[i], NY[count_N], coor[i%2], fixed_coor[i])
            count_N += 1
            if boundaryconditions[i-1] == 2:
                K[border[i][np.arange(0, 6)] - 1, :] = 0
                K[:, border[i][np.arange(0, 6)] - 1] = 0

                if isthereM != 0:
                    M[border[i][np.arange(0, 6)] - 1, :] = 0
                    M[:, border[i][np.arange(0, 6)] - 1] = 0
                    M[np.ix_(border[i][np.arange(0, 6)] - 1, border[i][np.arange(0, 6)] - 1)] = np.eye(6)

                F[border[i][np.arange(0, 6)] - 1] = 0
                K[np.ix_(border[i][np.arange(0, 6)] - 1, border[i][np.arange(0, 6)] - 1)] = np.eye(6)

        elif boundaryconditions[i] == 2:
            K[border[i]-1,:]=0
            K[:, border[i]-1]=0

            if isthereM != 0:

              M[border[i]-1,:]=0
              M[:, border[i]-1]=0
              M[np.ix_(border[i] - 1, border[i] - 1)] = np.eye(border[i].size)

            F[border[i]-1] = 0
            K[np.ix_(border[i]-1, border[i]-1)] = np.eye(border[i].size)


        elif boundaryconditions[i] == 3:
          F[border[i][np.arange(3, border[i].size, 6)] - 1, :] += BCMAssembly(X, T, borderss[i], Mben[count_Mom], coor[i%2], fixed_coor[i], coor[i%2-1])
          F[border[i][np.arange(4, border[i].size, 6)] - 1, :] += BCMAssembly(X, T, borderss[i], Mtor[count_Mom], coor[i%2], fixed_coor[i], coor[i%2])
          count_Mom += 1
          if boundaryconditions[i-1] == 2:
              K[border[i-1] - 1, :] = 0
              K[:, border[i-1] - 1] = 0
              F[border[i-1] - 1] = 0
              K[np.ix_(border[i-1] - 1, border[i-1] - 1)] = np.eye(border[i-1].size)

              if isthereM != 0:
                  M[border[i-1] - 1, :] = 0
                  M[:, border[i-1] - 1] = 0
                  M[np.ix_(border[i-1] - 1, border[i-1] - 1)] = np.eye(border[i-1].size)


        elif boundaryconditions[i] == 4:
          srch=np.nonzero(ENFRCDS2[i, :])
          for c in srch[0]:
           K[border[i][np.arange(c,border[i].size,6)]-1,:] = 0
           if isthereM != 0:
               M[border[i][np.arange(c,border[i].size,6)]-1,:] = 0
               M[np.ix_(border[i][np.arange(c, border[i].size, 6)] - 1, border[i][np.arange(c, border[i].size, 6)] - 1)] -= diag_mat(M[np.ix_(border[i][np.arange(c,border[i].size,6)]-1, border[i][np.arange(c,border[i].size,6)]-1)],analysis_type[0,2]) - np.eye(border[i][np.arange(c,border[i].size,6)].size)

           F[border[i][np.arange(c,border[i].size,6)]-1]=ENFRCDS1[i, c]
           #K[np.ix_(border1[np.arange(c,border1.size,6)]-1, border1[np.arange(c,border1.size,6)]-1)]=np.eye(border1[np.arange(c,border1.size,6)].size)
           K[np.ix_(border[i][np.arange(c, border[i].size, 6)] - 1, border[i][np.arange(c, border[i].size, 6)] - 1)] -= diag_mat(K[np.ix_(border[i][np.arange(c,border[i].size,6)]-1, border[i][np.arange(c,border[i].size,6)]-1)],analysis_type[0,2]) - np.eye(border[i][np.arange(c,border[i].size,6)].size)

        elif boundaryconditions[i] == 5:
            K[border[i][::6] - 1, :] = 0
            K[:, border[i][::6] - 1] = 0
            if isthereM != 0:

                M[border[i][::6] - 1, :] = 0
                M[:, border[i][::6] - 1] = 0
                M[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
                M[border[i][1::6] - 1, :] = 0
                M[:, border[i][1::6] - 1] = 0
                M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                M[border[i][2::6] - 1, :] = 0
                M[:, border[i][2::6] - 1] = 0
                M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                M[border[i][3::6] - 1, :] = 0
                M[:, border[i][3::6] - 1] = 0
                M[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

            F[border[i][::6] - 1] = 0
            K[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
            K[border[i][1::6] - 1, :] = 0
            K[:, border[i][1::6] - 1] = 0
            F[border[i][1::6] - 1] = 0
            K[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
            K[border[i][2::6] - 1, :] = 0
            K[:, border[i][2::6] - 1] = 0
            F[border[i][2::6] - 1] = 0
            K[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
            K[border[i][3::6] - 1, :] = 0
            K[:, border[i][3::6] - 1] = 0
            F[border[i][3::6] - 1] = 0
            K[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

        elif boundaryconditions[i] == 6:
            K[border[i][1::6] - 1, :] = 0
            K[:, border[i][1::6] - 1] = 0
            if isthereM != 0:

                M[border[i][1::6] - 1, :] = 0
                M[:, border[i][1::6] - 1] = 0
                M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                M[border[i][2::6] - 1, :] = 0
                M[:, border[i][2::6] - 1] = 0
                M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                M[border[i][3::6] - 1, :] = 0
                M[:, border[i][3::6] - 1] = 0
                M[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

            F[border[i][1::6] - 1] = 0
            K[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
            K[border[i][2::6] - 1, :] = 0
            K[:, border[i][2::6] - 1] = 0
            F[border[i][2::6] - 1] = 0
            K[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
            K[border[i][3::6] - 1, :] = 0
            K[:, border[i][3::6] - 1] = 0
            F[border[i][3::6] - 1] = 0
            K[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

        elif boundaryconditions[i] == 7:
            K[border[i][1::6] - 1, :] = 0
            K[:, border[i][1::6] - 1] = 0

            if isthereM != 0:

                M[border[i][1::6] - 1, :] = 0
                M[:, border[i][1::6] - 1] = 0
                M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                M[border[i][::6] - 1, :] = 0
                M[:, border[i][::6] - 1] = 0
                M[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
                M[border[i][4::6] - 1, :] = 0
                M[:, border[i][4::6] - 1] = 0
                M[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

            F[border[i][1::6] - 1] = 0
            K[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
            K[border[i][::6] - 1, :] = 0
            K[:, border[i][::6] - 1] = 0
            F[border[i][::6] - 1] = 0
            K[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
            K[border[i][4::6] - 1, :] = 0
            K[:, border[i][4::6] - 1] = 0
            F[border[i][4::6] - 1] = 0
            K[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

        elif boundaryconditions[i] == 8:
            K[border[i][1::6] - 1, :] = 0
            K[:, border[i][1::6] - 1] = 0

            if isthereM != 0:

                M[border[i][1::6] - 1, :] = 0
                M[:, border[i][1::6] - 1] = 0
                M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                M[border[i][::6] - 1, :] = 0
                M[:, border[i][::6] - 1] = 0
                M[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
                M[border[i][2::6] - 1, :] = 0
                M[:, border[i][2::6] - 1] = 0
                M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                M[border[i][4::6] - 1, :] = 0
                M[:, border[i][4::6] - 1] = 0
                M[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

            F[border[i][1::6] - 1] = 0
            K[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
            K[border[i][::6] - 1, :] = 0
            K[:, border[i][::6] - 1] = 0
            F[border[i][::6] - 1] = 0
            K[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
            K[border[i][2::6] - 1, :] = 0
            K[:, border[i][2::6] - 1] = 0
            F[border[i][2::6] - 1] = 0
            K[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
            K[border[i][4::6] - 1, :] = 0
            K[:, border[i][4::6] - 1] = 0
            F[border[i][4::6] - 1] = 0
            K[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

        i+=1

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

        NX = boundary_load['NX']
        NY = boundary_load['NY']
        Mben = boundary_load['Mbending']
        Mtor = boundary_load['Mtorsion']
        boundaryconditions = boundary_load['boundaryconditions']
        ENFRCDS = boundary_load['ENFRCDS']

        coor = [2, 1]

        ENFRCDS1 = np.array([ENFRCDS[:, np.arange(0, 5)]])
        ENFRCDS2 = np.array([ENFRCDS[:, np.arange(5, 10)]])
        ENFRCDS1 = ENFRCDS1[0]/Nincr

        ENFRCDS2 = ENFRCDS2[0]
        x1 = box[0, 0]
        y1 = box[0, 1]
        x2 = box[1, 0]
        y2 = box[1, 1]
        fixed_coor = box.flatten()

        borderss, border, border_size, everyborder = get_boundaries(X, b, nborders)

        i = 0
        count_N = 0
        count_Mom = 0
        while i < nborders:

            if boundaryconditions[i] == 1:
                F[border[i][np.arange(0, border[i].size, 6)] - 1] += BCAssembly(X, T, borderss[i], NX[i], coor[i%2], fixed_coor[i])
                F[border[i][np.arange(1, border[i].size, 6)] - 1] += BCAssembly(X, T, borderss[i], NY[i], coor[i%2], fixed_coor[i])
                count_N += 1

                if boundaryconditions[i-1] == 2:

                    if isthereM != 0:
                        M[border[i][np.arange(0, 6)] - 1, :] = 0
                        M[:, border[i][np.arange(0, 6)] - 1] = 0
                        M[np.ix_(border[i][np.arange(0, 6)] - 1, border[i][np.arange(0, 6)] - 1)] = np.eye(6)

                    F[border[i][np.arange(0, 6)] - 1] = 0

            elif boundaryconditions[i] == 2:

                if isthereM != 0:
                    M[border[i] - 1, :] = 0
                    M[:, border[i] - 1] = 0
                    M[np.ix_(border[i] - 1, border[i] - 1)] = np.eye(border[i].size)

                F[border[i] - 1] = 0


            elif boundaryconditions[i] == 3:
                F[border[i][np.arange(3, border[i].size, 6)] - 1, :] += BCMAssembly(X, T, borderss[i], Mben[i], coor[i%2], fixed_coor[i], coor[i%2-1])
                F[border[i][np.arange(4, border[i].size, 6)] - 1, :] += BCMAssembly(X, T, borderss[i], Mtor[i], coor[i%2], fixed_coor[i], coor[i%2])
                count_Mom += 1

                if boundaryconditions[i-1] == 2:

                    F[border[i-1] - 1] = 0

                    if isthereM != 0:
                        M[border[i-1] - 1, :] = 0
                        M[:, border[i-1] - 1] = 0
                        M[np.ix_(border[i-1] - 1, border[i-1] - 1)] = np.eye(border[i-1].size)


            elif boundaryconditions[i] == 5:

                if isthereM != 0:
                    M[border[i][::6] - 1, :] = 0
                    M[:, border[i][::6] - 1] = 0
                    M[np.ix_(border[i][::6] - 1, border[i][i][::6] - 1)] = np.eye(border[i][::6].size)
                    M[border[i][1::6] - 1, :] = 0
                    M[:, border[i][1::6] - 1] = 0
                    M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                    M[border[i][2::6] - 1, :] = 0
                    M[:, border[i][2::6] - 1] = 0
                    M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                    M[border[i][3::6] - 1, :] = 0
                    M[:, border[i][3::6] - 1] = 0
                    M[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

                F[border[i][::6] - 1] = 0

                F[border[i][1::6] - 1] = 0

                F[border[i][2::6] - 1] = 0

                F[border[i][3::6] - 1] = 0

            elif boundaryconditions[i] == 6:

                if isthereM != 0:
                    M[border[i][1::6] - 1, :] = 0
                    M[:, border[i][1::6] - 1] = 0
                    M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                    M[border[i][2::6] - 1, :] = 0
                    M[:, border[i][2::6] - 1] = 0
                    M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                    M[border[i][3::6] - 1, :] = 0
                    M[:, border[i][3::6] - 1] = 0
                    M[np.ix_(border[i][3::6] - 1, border[i][3::6] - 1)] = np.eye(border[i][3::6].size)

                F[border[i][1::6] - 1] = 0

                F[border[i][2::6] - 1] = 0

                F[border[i][3::6] - 1] = 0

            elif boundaryconditions[i] == 7:


                if isthereM != 0:
                    M[border[i][1::6] - 1, :] = 0
                    M[:, border[i][1::6] - 1] = 0
                    M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                    M[border[i][::6] - 1, :] = 0
                    M[:, border[i][::6] - 1] = 0
                    M[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
                    M[border[i][4::6] - 1, :] = 0
                    M[:, border[i][4::6] - 1] = 0
                    M[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

                F[border[i][1::6] - 1] = 0

                F[border[i][::6] - 1] = 0

                F[border[i][4::6] - 1] = 0

            elif boundaryconditions[i] == 8:


                if isthereM != 0:
                    M[border[i][1::6] - 1, :] = 0
                    M[:, border[i][1::6] - 1] = 0
                    M[np.ix_(border[i][1::6] - 1, border[i][1::6] - 1)] = np.eye(border[i][1::6].size)
                    M[border[i][::6] - 1, :] = 0
                    M[:, border[i][::6] - 1] = 0
                    M[np.ix_(border[i][::6] - 1, border[i][::6] - 1)] = np.eye(border[i][::6].size)
                    M[border[i][2::6] - 1, :] = 0
                    M[:, border[i][2::6] - 1] = 0
                    M[np.ix_(border[i][2::6] - 1, border[i][2::6] - 1)] = np.eye(border[i][2::6].size)
                    M[border[i][4::6] - 1, :] = 0
                    M[:, border[i][4::6] - 1] = 0
                    M[np.ix_(border[i][4::6] - 1, border[i][4::6] - 1)] = np.eye(border[i][4::6].size)

                F[border[i][1::6] - 1] = 0

                F[border[i][::6] - 1] = 0

                F[border[i][2::6] - 1] = 0

                F[border[i][4::6] - 1] = 0

            i += 1


        return F


def get_boundaries(X, b, nborders):

    borderss = []
    ind = 0
    while ind < nborders:
        i = 0
        while i < b.shape[1]:
            if b[ind, i] == 0:
                borderss.append(b[ind, np.arange(0, i)])
                break
            elif b[ind, i] != 0 and i == b.shape[1] - 1:
                borderss.append(b[ind, np.arange(0, i + 1)])
                break
            i += 1
        ind += 1

    ind = 0
    while ind < nborders:
        ii = 0
        while ii < borderss[ind].size:
            min = ii
            kk = ii + 1
            while kk < borderss[ind].size:
                if X[borderss[ind][kk] - 1, 1] < X[borderss[ind][min] - 1, 1]:
                    min = kk

                kk += 1
            rempl = borderss[ind][min]
            borderss[ind][min] = borderss[ind][ii]
            borderss[ind][ii] = rempl
            ii += 1

        ind += 1

    border = []
    ind = 0
    while ind < nborders:
        interm_border = np.zeros((1, 6 * borderss[ind].size))
        ii = 0
        while ii < borderss[ind].size:
            interm_border[0, np.arange(6 * ii, 6 * ii + 6)] = np.arange(6 * borderss[ind][ii] - 5,
                                                                        6 * borderss[ind][ii] + 1)
            ii += 1
        interm_border = interm_border.astype(int)
        interm_border = interm_border[0]

        border.append(interm_border)
        ind += 1

    border_size = 0
    i = 0
    while i < len(border):
        if border[i].shape[0] > border_size:
            border_size = border[i].shape[0]
        i += 1

    everyborder = np.zeros((nborders, border_size))
    i = 0
    while i < everyborder.shape[0]:
        everyborder[i, :] = np.concatenate((border[i], np.zeros((1, border_size - border[i].shape[0]))), axis=None)
        i += 1

    everyborder = everyborder.astype(int)


    return borderss, border, border_size, everyborder

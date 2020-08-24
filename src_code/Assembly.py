# -*-coding:Latin-1 -*

from core import *
from loading import *
from boundary_conditions import *

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

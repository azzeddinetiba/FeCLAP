# -*-coding:Latin-1 -*

from src_code.Assembly import *
from scipy.linalg import eig
import scipy.sparse.linalg
from src_code.postProc_calc import *
from src_code.inputting import *
import os
import NonLinearModule
import src_code.core

def FEM(total_loading,X,T,b,Ngauss,box,analysis_type,transient,material_param, *args):


    Klaw = material_param['constitutive_law']
    pho = material_param['pho']
    thickness = material_param['thickness']

    surface_nodal_load = total_loading['surf_node']
    surface_load = surface_nodal_load['surf']

    Xgauss, Wgauss = Quadrature(1, Ngauss)
    gp= Gauss3n(Ngauss)
    gp = gp[0]

    K, M, F = Assembly2D(X,T,surface_load,Wgauss,gp,Ngauss,Klaw,pho,thickness,analysis_type)

    fixed_borders, K, M, F = applying_BC(total_loading,X,T,b,box,K,F, analysis_type,M)



    if analysis_type[0,0] == 1:

        if analysis_type[0,1] == 1:

            if analysis_type[0,2] == 1:

                Fb = F.T
                Fb = Fb[0]

                U = np.linalg.solve(K, Fb)

            else:

                K = sp.csc_matrix(K)
                F= sp.csc_matrix(F)

                U = sp.linalg.spsolve(K, F)

            return K, F, U
        else:

            istherePlast = len(args)
            if istherePlast != 0:
                plast_param = args[0]

            Fb = F.T
            Fb = Fb[0]

            U, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU = \
                plastic_analysis(X, T, K, Fb, plast_param, material_param, b, box, total_loading, Ngauss)
            return  U, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU



    elif analysis_type[0,0] == 3:


        shape_for_index = M.shape[0]

        if analysis_type[0,2] == 1:
            Mb = np.delete(M, fixed_borders - 1, 0)
            Mb = np.delete(Mb, fixed_borders - 1, 1)
            Kb = np.delete(K, fixed_borders - 1, 0)
            Kb = np.delete(Kb, fixed_borders - 1, 1)

            modes_number = Mb.shape[0]

            w, modes = eig(Kb, Mb)

        else:

            boundary_load = total_loading['Bc']
            modes_number = boundary_load['modes']

            M = delete_row_csr(M, fixed_borders - 1)
            M = sp.csc_matrix.transpose(M)
            M = delete_row_csr(M, fixed_borders - 1)
            M = sp.csc_matrix.transpose(M)

            K = delete_row_csr(K, fixed_borders - 1)
            K = sp.csc_matrix.transpose(K)
            K = delete_row_csr(K, fixed_borders - 1)
            K = sp.csc_matrix.transpose(K)

            Kb = K
            Mb = M


            w, modes = sp.linalg.eigsh(Kb, modes_number , Mb, which = 'LM',  tol=1E-3, sigma = 0)

        fixed_borders = np.unique(fixed_borders)
        xxx = np.arange(0, shape_for_index)
        i = 0
        while i < fixed_borders.shape[0]:
            if i == 0:
                index = np.argwhere(xxx == fixed_borders[i] - 1)
            else:
                index = np.argwhere(yyy == fixed_borders[i] - 1)
            if i == 0:
                yyy = np.delete(xxx, index)
            else:
                yyy = np.delete(yyy, index)

            i += 1

        modal_indexes = yyy

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


            SOL[0:n_dof, 0] = x
            SOL[n_dof:2 * n_dof, 0] = xdot
            SOL[2 * n_dof: 3 * n_dof, 0] = xtwodots

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


def plastic_analysis(X, T, globalK, Fb, plast_param, material_param, b, box, total_loading, Ngauss):

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(script_dir, 'Results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    text_file = open(results_dir+'Non_Linear_Results.txt', "w")
    data_text = 'The yield stresses, displacements, and information on load increments are shown here:'


    TT=T.shape[0]

    gone_plastic = 0

    Xgauss, Wgauss = Quadrature(1, Ngauss)
    gp= Gauss3n(Ngauss)
    gp = gp[0]

    Q = material_param['Q_prime']
    pos = material_param['position']
    angles = material_param['angles']
    thickness = material_param['thickness']

    Nplies = len(angles)
    Nn = 6*X.shape[0]
    Nincr = plast_param['increments']
    Niter = plast_param['iterations']
    limit = plast_param['yield_limit']


    i=0
    while i<Nplies:
        if pos[i]*(pos[i]+thickness[i])<=0:
            mid_lay = i
            break
        i +=1

    #The load is divided on equal incremental loads
    deltaFb = Fb/Nincr
    Fb = np.zeros((deltaFb.shape[0],1))
    Fb = Fb.T
    Fb = Fb[0]
    saved_residual = []
    sxx = []
    syy = []
    sxy = []
    epxx = []
    epyy = []
    epxy = []

    saved_deltaU = []

    U = np.zeros((deltaFb.shape[0],1))
    U = U.T
    U = U[0]

    saved_strain_xx = np.zeros((TT, 1))
    saved_strain_yy = np.zeros((TT, 1))
    saved_strain_xy = np.zeros((TT, 1))

    saved_stress_xx = np.zeros((3, TT, Nplies))
    saved_stress_yy = np.zeros((3, TT, Nplies))
    saved_stress_xy = np.zeros((3, TT, Nplies))

    saved_stress_xx_ev = np.zeros((3, TT, Nplies))
    saved_stress_yy_ev = np.zeros((3, TT, Nplies))
    saved_stress_xy_ev = np.zeros((3, TT, Nplies))

    saved_delta_lambda = np.zeros((3,TT, Nplies))


    q = np.zeros((Nn, 1))
    q = applying_Fix_q(total_loading, X, T, b, box, q)
    q = q.T
    q = q[0]

    i=0
    while i<Nincr:


        #if i!=0:

        #    globalK, Q_data = tangent_stiffness(X, T, material_param, limit, b, box, total_loading, np.zeros((deltaFb.shape[0],0)), saved_stress_xx_ev, saved_stress_yy_ev, saved_stress_xy_ev, saved_delta_lambda, gone_plastic)



        deltaU = np.zeros((deltaFb.shape[0],1))
        deltaU = deltaU.T
        deltaU = deltaU[0]

        count=0
        count2 = 0

        ii=0
        while ii < Niter:

            data_tmp = '\nIncrement : ' + str(i+1) +'\niteration : '+str(ii)
            data_text += data_tmp
            print(data_tmp)

            u = deltaU[0::6]
            v = deltaU[1::6]
            w = deltaU[2::6]
            thetax =  deltaU[4::6]
            thetay = - deltaU[3::6]

            strain = strain_calc(X,T,u,v)


            if (ii != 0):

                q = np.zeros((Nn, 1))

                k = 0
                while k<TT:

                    Tie = T[k, :] + 1
                    Tie1 = np.concatenate((np.arange(6 * Tie[0] - 5, 6 * Tie[0] + 1),
                                           np.arange(6 * Tie[1] - 5, 6 * Tie[1] + 1),
                                           np.arange(6 * Tie[2] - 5, 6 * Tie[2] + 1)), 0)


                    j=0
                    while j<Nplies:


                        thick = 0
                        while thick < 3:


                            theta = angles[j]

                            Tr = np.array([[m.cos(theta) ** 2,              m.sin(theta) ** 2,              2 * m.sin(theta) * m.cos(theta)],
                                           [m.sin(theta) ** 2,              m.cos(theta) ** 2,              -2 * m.sin(theta) * m.cos(theta)],
                                           [-m.sin(theta) * m.cos(theta),   m.sin(theta) * m.cos(theta),   (m.cos(theta) ** 2) - m.sin(theta) ** 2]])

                            th = pos[j] + thick * thickness[j]/2

                            kappa = k_calc(X, T, w, thetax, thetay, k)

                            l=0

                            deltaStrain_used = np.zeros((1,3))
                            deltaStrain_used = deltaStrain_used[0]
                            while l<3:

                                laySTRAINxx = strain[k] + th * kappa[3*l]
                                laySTRAINxx = laySTRAINxx[0]

                                laySTRAINyy = strain[k + TT] + th * kappa[3*l + 1]
                                laySTRAINyy = laySTRAINyy[0]

                                laySTRAINxy = strain[k + 2 * TT] + th * kappa[3*l + 2]
                                laySTRAINxy = laySTRAINxy[0]

                                deltaStrain_used += (1/3) * np.array([laySTRAINxx, laySTRAINyy, laySTRAINxy])

                                l+=1





                            previous_stress = np.array([saved_stress_xx_ev[thick, k, j], saved_stress_yy_ev[thick, k, j], saved_stress_xy_ev[thick, k, j]])

                            Q_used = Q[np.arange(3*j,3*j+3), :]


                            rtrnAlg = NonLinearModule.returnAlg(previous_stress, Tr, limit, Q_used, deltaStrain_used)

                            interm = rtrnAlg.reshape((1, 4))
                            interm = np.array(interm[0])

                            previous_stress = interm[0:3]

                            if (interm[3]>1e-7 and count2 ==0):
                                data_tmp = '\nPlasticity occurring in the layer '+str(j+1)+', spotted at the element number, '+str(k)
                                data_text += data_tmp
                                print(data_tmp)
                                count2+=1

                            saved_delta_lambda[thick, k,j] = interm[3]

                            saved_strain_xx[k,0] = deltaStrain_used[0]
                            saved_strain_yy[k,0] = deltaStrain_used[1]
                            saved_strain_xy[k,0] = deltaStrain_used[2]


                            saved_stress_xx[thick, k,j] = previous_stress[0]
                            saved_stress_yy[thick, k,j] = previous_stress[1]
                            saved_stress_xy[thick, k,j] = previous_stress[2]



                            #stress = np.array([[saved_stress_xx[k,j]*thickness[j][0]],[saved_stress_xy[k,j]*thickness[j][0]],[saved_stress_xy[k,j]*thickness[j][0]],\
                            #                   [saved_stress_xx[k,j]*0.5 * (hi2 ** 2 - hi1 ** 2)[0]],[saved_stress_xy[k,j]*0.5 * (hi2 ** 2 - hi1 ** 2)[0]],[saved_stress_xy[k,j]*0.5 * (hi2 ** 2 - hi1 ** 2)[0]]])
                            #qe += internal_force_elem(X, T, k, gp, Wgauss, stress)

                            thick+=1

                        j+=1

                    qe = internal_force_elem(X, T, k, gp, Wgauss, saved_stress_xx,saved_stress_yy,saved_stress_xy,thickness,mid_lay,pos)
                    q[Tie1 - 1, :] += qe
                    k+=1


                q = applying_Fix_q(total_loading, X, T, b, box, q)
                q = q.T
                q = q[0]
                #q = np.dot(globalK, U)


            check_yield = saved_delta_lambda[saved_delta_lambda>1e-7]
            if check_yield.shape[0]!=0:
                gone_plastic = 1



            if gone_plastic==1:
                data_tmp = '\nComputing the new stiffness matrix'
                data_text += data_tmp
                print(data_tmp)
                globalK, Q_data = tangent_stiffness(X, T, material_param, limit, b, box, total_loading, np.zeros((deltaFb.shape[0],0)), saved_stress_xx, saved_stress_yy, saved_stress_xy, saved_delta_lambda, gone_plastic)
                count+=1

            residual = Fb + deltaFb - q
            residual = applying_Fix_q(total_loading, X, T, b, box, residual)

            tolerance = np.linalg.norm(residual)/np.linalg.norm(Fb+deltaFb)
            data_tmp = '\nResidual: '+str(tolerance)+' \n'
            data_text += data_tmp
            print(data_tmp)



            if ( tolerance < 0.001 )  or ( count>1 and tolerance < 0.06 ):


                saved_residual.append(residual)


                saved_stress_xx_ev = saved_stress_xx
                saved_stress_yy_ev = saved_stress_yy
                saved_stress_xy_ev = saved_stress_xy


                sxx.append(saved_stress_xx_ev)
                syy.append(saved_stress_yy_ev)
                sxy.append(saved_stress_xy_ev)


                epxx.append(saved_strain_xx)
                epyy.append(saved_strain_yy)
                epxy.append(saved_strain_xy)

                saved_deltaU.append(deltaU)

                break

            else:

                dU = np.linalg.solve(globalK,residual)

                deltaU = deltaU + dU

            ii+=1


        U = U + deltaU
        Fb = Fb + deltaFb

        i+=1



    n = text_file.write(data_text)
    text_file.close()

    return U, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU

def tangent_stiffness(X,T, material_param, limit, b, box, total_loading, F, saved_stress_xx, saved_stress_yy, saved_stress_xy, saved_delta_lambda, gone_plastic):


    Xgauss, Wgauss = Quadrature(1, 3)
    gp= Gauss3n(3)
    gp = gp[0]


    Q = material_param['Q_prime']
    pos = material_param['position']
    angles = material_param['angles']
    thickness = material_param['thickness']

    globalK = np.zeros((6*X.shape[0],6*X.shape[0]))

    nPlies = thickness.shape[0]
    nT = T.shape[0]


    Q_data = np.zeros((nT,nPlies,3,3))


    k=0
    while k<nT:

        Tie = T[k, :] + 1
        Tie1 = np.concatenate((np.arange(6 * Tie[0] - 5, 6 * Tie[0] + 1), np.arange(6 * Tie[1] - 5, 6 * Tie[1] + 1),
                               np.arange(6 * Tie[2] - 5, 6 * Tie[2] + 1)), 0)

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        j=0
        while j<nPlies:

            theta = angles[j]

            Tr = np.array([[m.cos(theta) ** 2,              m.sin(theta) ** 2,                       2 * m.sin(theta) * m.cos(theta)],
                           [m.sin(theta) ** 2,              m.cos(theta) ** 2,                       -2 * m.sin(theta) * m.cos(theta)],
                           [-m.sin(theta) * m.cos(theta),   m.sin(theta) * m.cos(theta),            (m.cos(theta) ** 2) - m.sin(theta) ** 2]])

            Q_used = Q[np.arange(3 * j, 3 * j + 3), :]


            thick=0
            while thick<3:
                stress_used = np.array([saved_stress_xx[thick,k,j],saved_stress_yy[thick,k,j],saved_stress_xy[thick,k,j],saved_delta_lambda[thick,k,j]])


                new_Q = NonLinearModule.Ktangent(stress_used, Tr, limit, Q_used)



                if gone_plastic == 1:
                    Q_data[k,j,:,:] = new_Q



                if thick!=1:
                    A = A + (thickness[j][0]/6) * new_Q
                    B = B + (thickness[j][0]/6) * (pos[j][0] + thick * 0.5 * thickness[j][0]) * new_Q
                    D = D + (thickness[j][0]/6) * ((pos[j][0] + thick * 0.5 * thickness[j][0])**2) * new_Q
                else:
                    A = A + 4 * (thickness[j][0]/6) * new_Q
                    B = B + 4 * (thickness[j][0]/6) * (pos[j][0] + thick * 0.5 * thickness[j][0]) * new_Q
                    D = D + 4 * (thickness[j][0]/6) * ((pos[j][0] + thick * 0.5 * thickness[j][0])**2) * new_Q

                thick+=1

            j+=1


        K = np.array([[A[0, 0], A[0, 1], A[0, 2], -B[0, 0], -B[0, 1], -B[0, 2]],
                          [A[1, 0], A[1, 1], A[1, 2], -B[1, 0], -B[1, 1], -B[1, 2]],
                          [A[2, 0], A[2, 1], A[2, 2], -B[2, 0], -B[2, 1], -B[2, 2]],
                          [-B[0, 0], -B[0, 1], -B[0, 2], D[0, 0], D[0, 1], D[0, 2]],
                          [-B[1, 0], -B[1, 1], -B[1, 2], D[1, 0], D[1, 1], D[1, 2]],
                          [-B[2, 0], -B[2, 1], -B[2, 2], D[2, 0], D[2, 1], D[2, 2]]])



        elemK = src_code.core.ElemMat(X,T,k,gp,Wgauss,K)
        globalK[np.ix_(Tie1 - 1, Tie1 - 1)] += elemK

        k+=1




    fixed_borders, globalK, M, F = applying_BC(total_loading,X,T,b,box,globalK,F)

    return globalK, Q_data
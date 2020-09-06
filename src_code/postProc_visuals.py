# -*-coding:Latin-1 -*

from src_code.postProc_calc import *
import matplotlib.pyplot as plt
import os
import math as m
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(script_dir, 'Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


def show_deformation(p,t,u, v, w, thetax, thetay):
    yes = int(input('Deformation ? Yes 1 / No 0 '))
    if yes == 1:
        component = int(input('exx 1 / eyy 2 / exy 3 ? '))
        th = int(input('Mid Plane 0 / at a thickness z (Oz) 1 ? '))
        if component == 1:
            titledef = 'Strain exx (Ox)'
        elif component == 2:
            titledef = 'Strain eyy (Oy)'
        else:
            titledef = 'Shear Gamma xy (Oxy)'

        if th == 0:
            STRAIN = strain_calc(p, t, u, v)
            nexx = int(STRAIN.size / 3)
            start = (component - 1) * nexx
            finish = component * nexx
            exx = STRAIN[start:finish]
            exx = exx.T
            exx = exx[0]
            exx = np.around(exx, decimals=10)
            fig6 = plt.figure()
            plt.gca().set_aspect('equal')
            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=exx, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig6.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')

        else:
            th = float(input('Thickness = '))
            thSTRAINxx, thSTRAINyy, thSTRAINxy = strain_calc_thick(p, t, u, v, w, thetax, thetay, th)
            fig7 = plt.figure()
            plt.gca().set_aspect('equal')
            if component == 1:
                levels = np.linspace(np.min(thSTRAINxx),np.max(thSTRAINxx),100)
                i=0
                while i<t.shape[0]:
                    exx = thSTRAINxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
            elif component == 2:
                levels = np.linspace(np.min(thSTRAINyy),np.max(thSTRAINyy),100)
                i=0
                while i<t.shape[0]:
                    eyy = thSTRAINyy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
            else:
                levels = np.linspace(np.min(thSTRAINxy),np.max(thSTRAINxy),100)
                i=0
                while i<t.shape[0]:
                    exy = thSTRAINxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1

            plt.colorbar()
            plt.title(titledef)
            fig7.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')


def show_stress(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness):
    yestr = int(input('Stress ? Yes 1 / No 0 '))
    if yestr == 1:
        component = int(input('Sxx 1 / Syy 2 / Sxy 3 ? '))
        th = float(input('Mid Plane 0 / at a thickness z (Oz) =  ? '))
        if component == 1:
            titledef = 'Stress Sxx (Ox)'
        elif component == 2:
            titledef = 'Stress Syy (Oy)'
        else:
            titledef = 'Stress xy (Oxy)'

        if th == 0:
            sxx, syy, sxy = stress_calc(p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness)

            fig8 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx, edgecolors='k')
            elif component == 2:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=syy, edgecolors='k')
            else:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxy, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig8.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')

        else:
            sxx, syy, sxy = stress_calc(p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness)
            fig9 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                levels = np.linspace(np.min(sxx),np.max(sxx),100)
                i=0
                while i<t.shape[0]:
                    exx = sxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1


            plt.colorbar()
            plt.title(titledef)
            fig9.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')


def show_displacement_thickness(p,u, v, thetax, thetay, pos, thickness, N):
    yes2 = int(input('plot Displacement/thickness at a point ? Yes 1 / No 0 '))
    if yes2 == 1:
        x = float(input('x coordinate of the point  ? '))
        y = float(input('y coordinate of the point  ? '))
        component = int(input('u (Ox) 1 / v(Oy) 2  ? '))

        yplot = [pos[N - 1], pos[0] + thickness[0]]

        pointu = 0
        distu = 10000

        i = 0
        while i < p.shape[0]:
            if m.sqrt(((p[i, 0] - x) ** 2) + ((p[i, 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[i, 0] - x) ** 2) + ((p[i, 1] - y) ** 2))
                pointu = i
            i += 1

        fig10 = plt.figure()
        if component == 1:
            xplot = [u[pointu] - pos[N - 1] * thetay[pointu], u[pointu] - (pos[0] + thickness[0]) * thetay[pointu]]
            titledef = 'Displacement u (Ox) by lmainate thickness '
            labeldef = 'u'
        elif component == 2:
            xplot = [v[pointu] - pos[N - 1] * thetax[pointu], v[pointu] - (pos[0] + thickness[0]) * thetax[pointu]]
            titledef = 'Displacement v (Oy) by lmainate thickness '
            labeldef = 'v'

        plt.plot(xplot, yplot, 'b-', label=labeldef)

        i = 0
        while i < N:
            plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
            i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig10.savefig(results_dir + titledef + ' .png')

        plt.show()
        print('press any key to continue')


def show_strain_thickness(p,t,u, v, w, thetax, thetay, pos, thickness, N):
    yes3 = int(input('plot Strain/thickness at an element ? Yes 1 / No 0 '))
    if yes3 == 1:

        thSTRAINxx1, thSTRAINyy1, thSTRAINxy1 = strain_calc_thick(p, t, u, v, w, thetax, thetay, pos[N - 1])
        thSTRAINxx2, thSTRAINyy2, thSTRAINxy2 = strain_calc_thick(p, t, u, v, w, thetax, thetay, pos[0] + thickness[0])
        x = float(input('x coordinate of a close point  ? '))
        y = float(input('y coordinate of a close point  ? '))
        component = int(input('exx (Ox) 1 / eyy(Oy) 2 / Gammaxy 3 (Oxy) ? '))

        yplot = [pos[N - 1], pos[0] + thickness[0]]

        pointu = 0
        distu = 10000

        i = 0
        while i < t.shape[0]:
            if x <= np.max(p[t[i,:],0]) and  np.min(p[t[i,:],0]) <= x and  np.min(p[t[i,:],1]) <= y and y <= np.max(p[t[i,:],1]):
                elementu = i
                break
            i += 1


        i = 0
        while i < 3:
            if m.sqrt(((p[t[elementu, i],0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[t[elementu, i], 0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2))
                pointu = i
            i += 1

        fig11 = plt.figure()
        if component == 1:
            xplot = [thSTRAINxx1[elementu,pointu], thSTRAINxx2[elementu,pointu]]
            titledef = 'Strain exx (Ox) by lmainate thickness '
            labeldef = 'exx'
        elif component == 2:
            xplot = [thSTRAINyy1[elementu,pointu], thSTRAINyy2[elementu,pointu]]
            titledef = 'Strain eyy (Oy) by lmainate thickness '
            labeldef = 'eyy'
        else:
            xplot = [thSTRAINxy1[elementu,pointu], thSTRAINxy2[elementu,pointu]]
            titledef = 'Shear Gammaxy (Oy) by lmainate thickness '
            labeldef = 'Gammaxy'

        plt.plot(xplot, yplot, 'b-', label=labeldef)

        i = 0
        while i < N:
            plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
            i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig11.savefig(results_dir + titledef + ' .png')

        plt.show()
        print('press any key to continue')


def show_stress_thickness(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness, N):
    yes4 = int(input('plot Stress/thickness at a point ? Yes 1 / No 0 '))
    if yes4 == 1:

        stressused = np.zeros((2 * N, 1))
        yplot = np.zeros((2 * N, 1))

        x = float(input('x coordinate of the point  ? '))
        y = float(input('y coordinate of the point  ? '))
        component = int(input('Sxx (Ox) 1 / Syy(Oy) 2 / Sxy 3 (Oxy) ? '))



        pointu = 0
        distu = 10000

        i = 0
        while i < t.shape[0]:
            if x <= np.max(p[t[i,:],0]) and  np.min(p[t[i,:],0]) <= x and  np.min(p[t[i,:],1]) <= y and y <= np.max(p[t[i,:],1]):
                elementu = i
                break
            i += 1


        i = 0
        while i < 3:
            if m.sqrt(((p[t[elementu, i],0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2)) <= distu:
                distu = m.sqrt(((p[t[elementu, i], 0] - x) ** 2) + ((p[t[elementu, i], 1] - y) ** 2))
                pointu = i
            i += 1


        fig11 = plt.figure()
        if component == 1:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)

                if pos[i]==0:

                    stressused = [sxx1[elementu], sxx2[elementu, pointu]]

                elif (pos[i]+thickness[i]) == 0:

                    stressused = [sxx1[elementu, pointu], sxx2[elementu]]

                else:

                    stressused = [sxx1[elementu, pointu], sxx2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Sxx (Ox) by lmainate thickness '
                plt.plot(xplot, yplot, 'b-')
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1

        elif component == 2:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)
                if pos[i] == 0:

                    stressused = [syy1[elementu], syy2[elementu, pointu]]

                elif (pos[i] + thickness[i]) == 0:

                    stressused = [syy1[elementu, pointu], syy2[elementu]]

                else:

                    stressused = [syy1[elementu, pointu], syy2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Syy (Oy) by lmainate thickness '
                plt.plot(xplot, yplot, 'b-')
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1


        else:

            i = 0
            while i < N:
                sxx1, syy1, sxy1 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i], pos, Qprime, thickness)
                sxx2, syy2, sxy2 = stress_calc(p, t, u, v, w, thetax, thetay, pos[i] + thickness[i], pos, Qprime, thickness)
                if pos[i] == 0:

                    stressused = [sxy1[elementu], sxy2[elementu, pointu]]

                elif (pos[i] + thickness[i]) == 0:

                    stressused = [sxy1[elementu, pointu], sxy2[elementu]]

                else:

                    stressused = [sxy1[elementu, pointu], sxy2[elementu, pointu]]

                xplot = stressused
                yplot = [pos[i], pos[i] + thickness[i]]
                titledef = 'Stress Sxy (Ox) by lmainate thickness '
                labeldef = 'Sxy'
                plt.plot(xplot, yplot, 'b-', label=labeldef)
                plt.axhline(y=pos[i], color='k', linestyle='-', label='ply n' + str(i + 1))
                i += 1

        plt.legend(loc='best')
        plt.title(titledef)
        fig11.savefig(results_dir + titledef + ' .png')

        plt.show()
        print('press any key to continue')


def show_deformation_LT(p,t,u, v, w, thetax, thetay,pos,thickness, angles):
    nT=t.shape[0]
    yes = int(input('Deformation in principal ply axes ? Yes 1 / No 0 '))
    if yes == 1:
        component = int(input('eLL 1 / eTT 2 / eLT 3 ? '))
        th = int(input('Mid Plane 0 / at a thickness z (Oz) 1 ? '))
        if component == 1:
            titledef = 'Strain eLL (Ox)'
        elif component == 2:
            titledef = 'Strain eTT (Oy)'
        else:
            titledef = 'Shear Gamma LT (Oxy)'

        if th == 0:
            strain_L, strain_T, strain_LT = strain_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,thickness, angles)
            strain_L, strain_T, strain_LT = np.reshape(strain_L,(nT,1)), np.reshape(strain_T,(nT,1)) , np.reshape(strain_LT,(nT,1))
            STRAIN = np.concatenate((strain_L, strain_T, strain_LT),axis=0)
            nexx = int(STRAIN.size / 3)
            start = (component - 1) * nexx
            finish = component * nexx
            exx = STRAIN[start:finish]
            exx = exx.T
            exx = exx[0]
            exx = np.around(exx, decimals=10)
            fig12 = plt.figure()
            plt.gca().set_aspect('equal')
            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=exx, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig12.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')

        else:
            th = float(input('Thickness = '))
            strain_L, strain_T, strain_LT = strain_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,thickness, angles)
            fig13 = plt.figure()

            if component == 1:
                levels = np.linspace(np.min(strain_L),np.max(strain_L),100)
                i=0
                while i<t.shape[0]:
                    exx = strain_L[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
            elif component == 2:
                levels = np.linspace(np.min(strain_T),np.max(strain_T),100)
                i=0
                while i<t.shape[0]:
                    eyy = strain_T[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
            else:
                levels = np.linspace(np.min(strain_LT),np.max(strain_LT),100)
                i=0
                while i<t.shape[0]:
                    exy = strain_LT[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1

            plt.colorbar()
            plt.title(titledef)
            fig13.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')


def show_stress_LT(p,t,u, v, w, thetax, thetay, pos, Qprime, thickness, angles):
    yestr = int(input('Stress in ply principal axes ? Yes 1 / No 0 '))
    if yestr == 1:
        component = int(input('S LL 1 / S TT 2 / S LT 3 ? '))
        th = float(input('Mid Plane 0 / at a thickness z (Oz) =  ? '))
        if component == 1:
            titledef = 'Stress S LL (Ox)'
        elif component == 2:
            titledef = 'Stress S TT (Oy)'
        else:
            titledef = 'Stress LT (Oxy)'

        if th == 0:
            sxx, syy, sxy = stress_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles)

            fig14 = plt.figure()
            plt.gca().set_aspect('equal')

            if component == 1:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx, edgecolors='k')
            elif component == 2:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=syy, edgecolors='k')
            else:
                plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxy, edgecolors='k')
            plt.colorbar()
            plt.title(titledef)
            fig14.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')

        else:
            sxx, syy, sxy = stress_LT_calc(p,t,u,v,w,thetax,thetay,th,pos,Qprime,thickness, angles)
            fig15 = plt.figure()

            if component == 1:
                levels = np.linspace(np.min(sxx),np.max(sxx),100)
                i=0
                while i<t.shape[0]:
                    exx = sxx[i,:]
                    exx = np.around(exx, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                    i+=1
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1


            plt.colorbar()
            plt.title(titledef)
            fig15.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')


def show_Hoffman_stress(p,t,u,v,w,thetax,thetay,pos,Qprime,thickness, angles):

    yeshoffstr = int(input('Hoffman Stress ? Yes 1 / No 0 '))


    if yeshoffstr == 1:

        th = float(input('thickness = ?'))
        SLLt = float(input('Longitudinal tension strength SLLt = '))
        SLLc = float(input('Longitudinal compression strength SLLc = '))
        STTt = float(input('Transversal tension strength STTt = '))
        STTc = float(input('Transversal compression strength STTc = '))
        SLT = float(input('Shear strength SLT = '))

        titledef = 'Hoffman criterion'

        Hoffman_stress = Hoffman(SLLt, SLLc, STTt, STTc, SLT, p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness, angles)

        if th == 0:

            fig16 = plt.figure()
            plt.gca().set_aspect('equal')

            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=Hoffman_stress, edgecolors='k')

            plt.colorbar()
            plt.title(titledef)
            fig16.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')

        else:
            fig17 = plt.figure()

            levels = np.linspace(np.min(Hoffman_stress), np.max(Hoffman_stress), 100)
            i = 0
            while i < t.shape[0]:
                exx = Hoffman_stress[i, :]
                exx = np.around(exx, decimals=10)
                plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exx, levels, cmap="jet")
                i += 1

            plt.colorbar()
            plt.title(titledef)
            fig17.savefig(results_dir + titledef + ' .png')

            plt.show()
            print('press any key to continue')


def animate_mode(frq,modes,mode_number,modal_indexes,mesh_size,p,t):

    num=500
    freq1 = frq[mode_number-1]
    if freq1!=0:
     t1 = np.linspace(0, 1, num) / freq1
    else:
     t1 = np.linspace(0, 1, num)

    vec = np.zeros((6*mesh_size,1))
    vec[modal_indexes,0] = modes[:,mode_number-1]
    vec = vec[2::6]
    normalise = np.trim_zeros(np.absolute(vec))
    if normalise.size !=0:
        vec = vec*(1/np.max(normalise))

    if freq1!=0:
     y = vec * np.sin(2*m.pi*t1*freq1)
    else:
     y = vec * np.sin(2 * m.pi * t1)


    fig = plt.figure(figsize=(10.2, 6.8), dpi=100)
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, np.max(y))
    ax.set_axis_off()

    def run_anim(i1):
        ax = fig.gca(projection='3d')
        ax.cla()
        ax.set_zlim3d(0, np.max(y))
        ax.set_axis_off()
        plt.title('Mode ' + str(mode_number) + ' Animation, Frequency ' + '%.2f' % freq1 + ' Hz')
        running_anim = ax.plot_trisurf(p[:,0], p[:,1], t,y[:,i1], cmap="jet", antialiased=True)

        return running_anim

    ani = FuncAnimation(fig, run_anim, frames=num, interval=500, repeat=True, blit=False)

    return ani


def animate_transient(SOL, p, t, delta_T, n_iter):

    num=int(n_iter)

    fig = plt.figure(figsize=(10.2, 6.8), dpi=100)
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, np.max(SOL))
    ax.set_axis_off()

    def run_anim(i1):
        ax = fig.gca(projection='3d')
        ax.cla()
        ax.set_zlim3d(0, np.max(SOL))
        instant = delta_T*i1
        plt.title('Transient response of the transverse displacement of the plate ' + 't = ' +  '%.2f' % instant + ' s' )
        running_anim = ax.plot_trisurf(p[:,0], p[:,1], t,SOL[2::6,i1], cmap="jet", antialiased=True)

        return running_anim

    ani = FuncAnimation(fig, run_anim, frames=num, interval=500, repeat=False, blit=False)

    return ani


def transient_postProc(transient, U, p, t):

    delta_T = transient['time_step']

    tf = transient['final_instant']

    n_iter = tf / delta_T

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    one_D_two_D = 3

    while one_D_two_D != 0:

        one_D_two_D = int(input('Plot for a point 1 or animate the plate 2, Exit 0 '))

        if one_D_two_D == 1:

            XY = np.zeros((1, 2))
            print('Point to display : [x y] ')
            for j in np.arange(0, 2):
                XY[0, j] = float(input())
            pointdisp = -1
            distdisp = 10000
            ii = 0
            while ii < p.shape[0]:
                if m.sqrt(((p[ii, 0] - XY[0, 0]) ** 2) + ((p[ii, 1] - XY[0, 1]) ** 2)) <= distdisp:
                    distdisp = m.sqrt(((p[ii, 0] - XY[0, 0]) ** 2) + ((p[ii, 1] - XY[0, 1]) ** 2))
                    pointdisp = ii
                ii += 1

            dof = int(input('Degree of freedom : 1/2/3/4/5'))

            fig6 = plt.figure()
            time_domain = np.linspace(0, tf, int(n_iter))
            if transient['scheme'] == 1:
                plt.plot(time_domain, U[6 * (pointdisp + 1) - 7 + dof, :])
            else:
                plt.plot(time_domain, U[6 * (pointdisp + 1) - 7 + dof, :])
            plt.grid()
            plt.xlabel('Time [s]')
            plt.ylabel('Displacement [m]')
            plt.title('Transient analysis')
            plt.savefig('Transient analysis' + '.png')

        elif one_D_two_D == 2:

            trans_anim = animate_transient(U, p, t, delta_T, n_iter)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            trans_anim.save(results_dir+'transient.mp4', writer=writer)


def General_Post_proc(U, p, t, material_param):


    thickness = material_param['thickness']
    angles = material_param ['angles']
    Qprime = material_param ['Q_prime']
    pos = material_param['angles']
    N = len(thickness)

    u = U[0::6]
    v = U[1::6]
    w = U[2::6]
    thetay = -U[3::6]
    thetax = U[4::6]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title('Transversal Displacement w (Oz) ')
    ax.plot_trisurf(p[:, 0], p[:, 1], t, w, linewidth=0.2, antialiased=True, cmap='jet')
    fig.savefig(results_dir + 'Transversal Displacement w (Oz) .png')
    plt.show()
    print('press any key to continue')

    print()

    fig2 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, u, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement u (Ox) ')
    fig2.savefig(results_dir + 'Displacement u (Ox) .png')
    plt.show()
    print('press any key to continue')

    fig3 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, v, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement v (Oy) ')
    fig3.savefig(results_dir + 'Displacement v (Oy) .png')
    plt.show()
    print('press any key to continue')

    fig4 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetax, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Ox) ')

    fig4.savefig(results_dir + 'Rotation on (Ox) .png')
    plt.show()
    print('press any key to continue')

    fig5 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetay, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Oy)')
    fig5.savefig(results_dir + 'Rotation on (Oy) .png')
    plt.show()
    print('press any key to continue')

    show_deformation(p, t, u, v, w, thetax, thetay)

    show_stress(p, t, u, v, w, thetax, thetay, pos, Qprime, thickness)

    show_displacement_thickness(p, u, v, thetax, thetay, pos, thickness, N)

    show_strain_thickness(p, t, u, v, w, thetax, thetay, pos, thickness, N)

    show_stress_thickness(p, t, u, v, w, thetax, thetay, pos, Qprime, thickness, N)

    show_deformation_LT(p, t, u, v, w, thetax, thetay, pos, thickness, angles)

    show_stress_LT(p, t, u, v, w, thetax, thetay, pos, Qprime, thickness, angles)

    show_Hoffman_stress(p, t, u, v, w, thetax, thetay, pos, Qprime, thickness, angles)

    print()


def Plastic_Post_proc(U, p, t, material_param, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU):


    chosen_step = int(input('Post Process at the end of Increment number : ?  '))

    st = 0
    resul = np.zeros((Fb.shape[0], 1))
    resul = resul.T
    resul = resul[0]
    while st < chosen_step:
        resul += saved_deltaU[st]
        st += 1

    u = resul[0::6]
    v = resul[1::6]
    w = resul[2::6]
    thetay = -resul[3::6]
    thetax = resul[4::6]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title('Transversal Displacement w (Oz) ')
    ax.plot_trisurf(p[:, 0], p[:, 1], t, w, linewidth=0.2, antialiased=True, cmap='jet')
    fig.savefig(results_dir + 'Transversal Displacement w (Oz) .png')
    plt.show()
    print('press any key to continue')

    print()

    fig2 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, u, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement u (Ox) ')
    fig2.savefig(results_dir + 'Displacement u (Ox) .png')
    plt.show()
    print('press any key to continue')

    fig3 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, v, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement v (Oy) ')
    fig3.savefig(results_dir + 'Displacement v (Oy) .png')
    plt.show()
    print('press any key to continue')

    fig4 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetax, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Ox) ')

    fig4.savefig(results_dir + 'Rotation on (Ox) .png')
    plt.show()
    print('press any key to continue')

    fig5 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetay, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Oy)')
    fig5.savefig(results_dir + 'Rotation on (Oy) .png')
    plt.show()
    print('press any key to continue')



    yestr = int(input('Stress ? Yes 1 / No 0 '))
    component = 1
    if yestr == 1:
        while component !=0 :
            component = int(input('Sxx 1 / Syy 2 / Sxy 3 / Exit 0? '))
            if component != 0:
                lay  = int(input('Layer number ? '))
                chosen_step = int(input('At the end of Increment number : ?  '))
                if component == 1:
                    titledef = 'Stress Sxx (Ox)'
                elif component == 2:
                    titledef = 'Stress Syy (Oy)'
                else:
                    titledef = 'Stress xy (Oxy)'


                fig9 = plt.figure()
                plt.gca().set_aspect('equal')

                if component == 1:
                    sxx_used = sxx[chosen_step]
                    sxx_used = sxx_used[1,:,lay-1]
                    plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx_used, edgecolors='k')
                elif component == 2:
                    syy_used = syy[chosen_step]
                    syy_used = syy_used[1,:,lay-1]
                    plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=syy_used, edgecolors='k')
                else:
                    sxy_used = sxy[chosen_step]
                    sxy_used = sxy_used[1,:,lay-1]
                    plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxy_used, edgecolors='k')
                plt.colorbar()
                plt.title(titledef)
                fig9.savefig(results_dir + titledef + ' .png')
                plt.show()
                print('press any key to continue')


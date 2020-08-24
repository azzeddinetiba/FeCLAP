# -*-coding:Latin-1 -*

from postProc_calc import *
import matplotlib.pyplot as plt
import os
import math as m
from matplotlib.animation import FuncAnimation

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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(thSTRAINyy),np.max(thSTRAINyy),100)
                i=0
                while i<t.shape[0]:
                    eyy = thSTRAINyy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(thSTRAINxy),np.max(thSTRAINxy),100)
                i=0
                while i<t.shape[0]:
                    exy = thSTRAINxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")

            plt.colorbar()
            plt.title(titledef)
            fig7.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")


            plt.colorbar()
            plt.title(titledef)
            fig9.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
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
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
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
        #img = cv2.imread(titledef + ' .png')
        #cv2.imshow(titledef, img)
        #cv2.waitKey(0)
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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(strain_T),np.max(strain_T),100)
                i=0
                while i<t.shape[0]:
                    eyy = strain_T[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(strain_LT),np.max(strain_LT),100)
                i=0
                while i<t.shape[0]:
                    exy = strain_LT[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")

            plt.colorbar()
            plt.title(titledef)
            fig13.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxx, 500, cmap="jet")
            elif component == 2:
                levels = np.linspace(np.min(syy),np.max(syy),100)
                i=0
                while i<t.shape[0]:
                    eyy = syy[i,:]
                    eyy = np.around(eyy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], eyy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINyy, 500, cmap="jet")
            else:
                levels = np.linspace(np.min(sxy),np.max(sxy),100)
                i=0
                while i<t.shape[0]:
                    exy = sxy[i,:]
                    exy = np.around(exy, decimals=10)
                    plt.tricontourf(p[t[i, :], 0], p[t[i, :], 1], exy, levels, cmap="jet")
                    i+=1
                    #plt.tricontourf(p[:, 0], p[:, 1], t, thSTRAINxy, 500, cmap="jet")


            plt.colorbar()
            plt.title(titledef)
            fig15.savefig(results_dir + titledef + ' .png')
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
            #img = cv2.imread(titledef + ' .png')
            #cv2.imshow(titledef, img)
            #cv2.waitKey(0)
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
        plt.title('Animation du mode ' + str(mode_number) + ' de frequence ' + '%.2f' % freq1 + ' Hz')
        running_anim = ax.plot_trisurf(p[:,0], p[:,1], t,y[:,i1], cmap="jet", antialiased=True)

        return running_anim

    ani = FuncAnimation(fig, run_anim, frames=num, interval=500, repeat=True, blit=False)
    #plt.show()

    return ani


def animate_transient(SOL, p, t, delta_T):

    num=500

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
    #plt.show()

    return ani

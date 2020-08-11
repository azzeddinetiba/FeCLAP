# -*- coding: utf-8 -*-
# -*-coding:Latin-1 -*

import os
import numpy as np
import math as m
import os
from Assembly_Sparse import *
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib as mpl
import scipy.sparse as sp

mpl.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
import cv2
import pylab
import scipy
from mpl_toolkits.mplot3d import Axes3D

print(
    'ENSAM CASABLANCA \nENSAM Sciences et Technologies\n\nA code for the linear static and modal analysis of laminate composite plates using the Kirchoff-Love theory of thin plates and the lmainate composites theory\nThe supported geometry is a rectangular plate with or without one hole\n\nTIBA Azzeddine  -  2020\n')

"""
xh=yh=0
x1=float(input('coordinate of the left border x1= '))
y1=float(input('coordinate of the lower border y1= '))
x2=float(input('coordinate of the right border x2= '))
y2=float(input('coordinate of the upper border y2= '))
h=float(input('Mesh Size : '))
radius=float(input('Add a hole Yes 1/No 0 : '))
if radius !=0:
    radius=float(input('The Radius : '))
    xh=float(input('x of the center of the hole: '))
    yh=float(input('y of the center of the hole: '))
"""

x1 = 0
y1 = 0
x2 = 1
y2 = 0.8
h = 0.03
radius = 0.1
xh = 0.2
yh = 0.3
box = np.array([[x1, y1], [x2, y2]])
Ngauss = 3
# Ngauss=int(input('Number of GAUSS Points :'))
# print()

print()
p, t, b = mesh(x1, x2, y1, y2, h, radius, xh, yh)


# N, PPT, angles, thickness, TH, pho=get_plies()
print()

N = 10
thickness = 0.00027 * np.ones((N, 1))
pho = 3500 * np.ones((N, 1))
angles = np.array([[90], [0], [45], [-45], [90], [90], [-45], [45], [0], [90]])
TH = sum(thickness) / 2
PPT = np.concatenate(
    (181e9 * np.ones((N, 1)), 10.3e9 * np.ones((N, 1)), 0.3 * np.ones((N, 1)), 7.17e9 * np.ones((N, 1))), axis=1)

pos, Q, Qprime, A, B, D, Klaw = constitutive_law(N, PPT, angles, thickness, TH)
print()

analysis_type = get_type()
print()

if analysis_type != 3:
    f, g, h, pointload, NODALLOAD = get_loads(p)
    print()

    NX1, NY1, NX2, NY2, NX3, NY3, NX4, NY4, MY1, MXY1, MX2, MXY2, MY3, MXY3, MX4, MXY4, boundaryconditions, ENFRCDS = get_boundaryconditions(
        analysis_type)
    print()

    transient = 0

if analysis_type != 3:

    if analysis_type == 2:
        transient = get_transient()

    t1 = time.time()

    Kb, F, U = FEM(f, g, h, NX1, NY1, NX2, NY2, NX3, NY3, NX4, NY4, MY1, MXY1, MX2, MXY2, MY3, MXY3, MX4, MXY4,
                   boundaryconditions, ENFRCDS, p, t, b, Ngauss, Klaw, box, pointload, NODALLOAD, pho, thickness,
                   analysis_type, transient)

    t2 = time.time()

else:

    NX1, NY1, NX2, NY2, NX3, NY3, NX4, NY4, MY1, MXY1, MX2, MXY2, MY3, MXY3, MX4, MXY4, boundaryconditions, ENFRCDS = get_boundaryconditions(
        analysis_type)
    transient = 0
    f = lambda x, y: 0
    g = lambda x, y: 0
    h = lambda x, y: 0
    pointload = ([])
    NODALLOAD = np.array([np.array([0, 0, 0])])
    NODALLOAD = NODALLOAD[1::, :]
    load = 1

    t1 = time.time()

    Kb, Mb, freq, modes, modal_indexes = FEM(f, g, h, NX1, NY1, NX2, NY2, NX3, NY3, NX4, NY4, MY1, MXY1, MX2, MXY2, MY3,
                                             MXY3, MX4, MXY4, boundaryconditions, ENFRCDS, p, t, b, Ngauss, Klaw, box,
                                             pointload, NODALLOAD, pho, thickness, analysis_type, transient)

    t2 = time.time()

    print(np.max(freq))
    mesh_size = p.shape[0]
    mode_number = 1
    while mode_number != 0:
        mode_number = int(input('Visualiser mode numero ?  /Exit 0 '))
        if mode_number == 0:
            break
        modal_anim = animate_mode(freq, modes, mode_number, modal_indexes, mesh_size, p, t)
        writervideo = animation.FFMpegWriter(fps=60)
        modal_anim.save('modal.mp4', writer=writervideo)

if analysis_type == 1:

    u = U[0::6]
    v = U[1::6]
    w = U[2::6]
    thetay = -U[3::6]
    thetax = U[4::6]

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.title('Transversal Displacement w (Oz) ')
    ax.plot_trisurf(p[:, 0], p[:, 1], t, w, linewidth=0.2, antialiased=True, cmap='jet')
    fig.savefig(results_dir + 'Transversal Displacement w (Oz) .png')
    # img = cv2.imread('Transversal Displacement w (Oz) .png')
    # cv2.imshow('Transversal Displacement w (Oz) ', img)
    # cv2.waitKey(0)
    plt.show()
    print('press any key to continue')

    print()

    fig2 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, u, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement u (Ox) ')
    fig2.savefig(results_dir + 'Displacement u (Ox) .png')
    # img = cv2.imread('Displacement u (Ox) .png')
    # cv2.imshow('Displacement u (Ox) ', img)
    # cv2.waitKey(0)
    plt.show()
    print('press any key to continue')

    fig3 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, v, 500, cmap="jet")
    plt.colorbar()
    plt.title('Displacement v (Oy) ')
    fig3.savefig(results_dir + 'Displacement v (Oy) .png')
    # img = cv2.imread('Displacement v (Oy) .png')
    # cv2.imshow('Displacement v (Oy)', img)
    # cv2.waitKey(0)
    plt.show()
    print('press any key to continue')

    fig4 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetax, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Ox) ')

    fig4.savefig(results_dir + 'Rotation on (Ox) .png')
    # img = cv2.imread('Rotation on (Ox) .png')
    # cv2.imshow('Rotation on (Ox) ', img)
    # cv2.waitKey(0)
    plt.show()
    print('press any key to continue')

    fig5 = plt.figure()
    plt.gca().set_aspect('equal')
    plt.tricontourf(p[:, 0], p[:, 1], t, thetay, 500, cmap="jet")
    plt.colorbar()
    plt.title('Rotation on (Oy)')
    fig5.savefig(results_dir + 'Rotation on (Oy) .png')
    # img = cv2.imread('Rotation on (Oy) .png')
    # cv2.imshow('Rotation on (Oy) ', img)
    # cv2.waitKey(0)
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

if analysis_type == 2:

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

            trans_anim = animate_transient(U, p, t, delta_T)
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            trans_anim.save('transient.mp4', writer=writer)

print('Time : ' + str(t2 - t1) + ' s')
os.system("pause")

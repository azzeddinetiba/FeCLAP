# -*- coding: utf-8 -*-
# -*-coding:Latin-1 -*


from src_code import *
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import src_code.postProc_visuals
mpl.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

print(
    '\nA small solver for the static (linear and perfectly plastic) modal and transient analysis of laminate composite plates using the Kirchoff-Love theory of thin plates and the lmainate composites theory\nThe supported geometry is a rectangular plate with or without one hole\n\nTIBA Azzeddine  -  2020\n')


# Allowing the user to input the geometry and materials
x1=float(input('coordinate of the left border x1= '))
y1=float(input('coordinate of the lower border y1= '))
x2=float(input('coordinate of the right border x2= '))
y2=float(input('coordinate of the upper border y2= '))
h=float(input('Mesh Size : '))
radius=float(input('Add a hole Yes 1/No 0 : '))
xh = 0
yh = 0
if radius !=0:
    radius=float(input('The Radius : '))
    xh=float(input('x of the center of the hole: '))
    yh=float(input('y of the center of the hole: '))
box = np.array([[x1, y1], [x2, y2]])


p, t, b = mesh(x1, x2, y1, y2, h, radius, xh, yh)


#analysis_type
analysis_type = get_type()
print()


# Constitutive law
N, PPT, angles, thickness, TH, pho=get_plies(analysis_type)
print()
pos, Q, Qprime, A, B, D, Klaw = constitutive_law(N, PPT, angles, thickness, TH)
print()


material_param = {'thickness': thickness, 'pho': pho, 'constitutive_law': Klaw, \
                  'Q_matrix': Q, 'angles': angles, 'position': pos, 'Q_prime':Qprime}

Ngauss=0
while (Ngauss!=3 and Ngauss!=4 and Ngauss!=7):
    Ngauss=int(input('Number of GAUSS Points : (3, 4 or 7) '))
    print()


script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


if analysis_type[0,0] != 3:
    surface_nodal_load = get_loads(p)
    print()

    boundary_load = get_boundaryconditions(
        analysis_type)
    print()

    total_loading = {'Bc':boundary_load, 'surf_node':surface_nodal_load}
    print()

    transient = 0

if analysis_type[0,0] != 3:

    if analysis_type[0,1] == 2:

        print('\n Warning: The Non Linear Module is still experimental for out-of plane loads')

        plast_param = get_plastic()


        t1 = time.time()

        U, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU = \
            FEM(total_loading, p, t, b, Ngauss, box,
                analysis_type, transient, material_param, plast_param)

        t2 = time.time()

        print('Finite element Solution finding Time : ' + str(t2 - t1) + ' s')


    else:

        if analysis_type[0,0] == 2:
            transient = get_transient()

        t1 = time.time()


        Kb, F, U = FEM(total_loading, p, t, b, Ngauss, box,
                       analysis_type, transient, material_param)

        t2 = time.time()

        print('Finite element Solution finding Time : ' + str(t2 - t1) + ' s')


else:

    boundary_load = get_boundaryconditions(
        analysis_type)
    transient = 0
    f = lambda x, y: 0
    g = lambda x, y: 0
    h = lambda x, y: 0
    surface_load = {'z':f, 'x':g, 'y':h}

    pointload = ([])
    NODALLOAD = np.array([np.array([0, 0, 0])])
    NODALLOAD = NODALLOAD[1::, :]
    nodal_load = {'coord':pointload, 'value':NODALLOAD}
    load = 1
    surface_nodal_load = {'surf':surface_load, 'node':nodal_load}

    total_loading = {'Bc':boundary_load, 'surf_node':surface_nodal_load}


    t1 = time.time()

    Kb, Mb, freq, modes, modal_indexes = FEM(total_loading, p, t, b, Ngauss, box,
                   analysis_type, transient, material_param)


    a_file = open(results_dir+'modal_frequencies.txt', "w")
    for row in np.array([freq]):
        np.savetxt(a_file, row)

    a_file.close()

    t2 = time.time()

    print('Finite element Solution finding Time : ' + str(t2 - t1) + ' s')

    mesh_size = p.shape[0]
    mode_number = 1
    while mode_number != 0:
        mode_number = int(input('Visualize Mode Number ?  /Exit 0 '))
        if mode_number == 0:
            break
        if mode_number > modes.shape[1] and analysis_type[0,2] == 0:
            print('The sparse solver only extracts the chosen number of modes,\n \
            Please choose another mode to visualize ')

            continue

        if mode_number > modes.shape[1] and analysis_type[0, 2] == 1:
            print('The chosen number exceeds the number of Dofs, there is no corresponding modes\n \
            Please choose another mode to visualize ')

            continue

        modal_anim = animate_mode(freq, modes, mode_number, modal_indexes, mesh_size, p, t)
        writervideo = animation.FFMpegWriter(fps=60)
        modal_anim.save(results_dir+"modal"+"_mode_"+str(mode_number)+".mp4", writer=writervideo)

if analysis_type[0,0] == 1:


    if analysis_type[0,1] == 2:

        src_code.postProc_visuals.Plastic_Post_proc(U, p, t, material_param, Fb, sxx, syy, sxy,\
                          saved_residual, epxx, epyy, epxy, saved_deltaU)


    else:

        src_code.postProc_visuals.General_Post_proc(U, p, t, material_param)


if analysis_type[0,0] == 2:

    src_code.transient_postProc(transient, U, p, t)

os.system("pause")

# -*- coding: utf-8 -*-
# -*-coding:Latin-1 -*


from src_code import *
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
import matplotlib as mpl
from src_code.postProc_visuals import *
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

print(
    '\nExample Cases For FeCLAP\n\nTIBA Azzeddine  -  2020\n')


#------------------------    CASE  1 Non linear Response  -----------------------------
#-------------------------------------------------------------------------------------
#The Geometry
xh=yh=0
x1 = 0
y1 = 0
x2 = 1
y2 = 0.8
h = 0.1
radius = 0
xh = 0.2
yh = 0.3
box = np.array([[x1, y1], [x2, y2]])

# Meshing the geometry
p, t, b = mesh(x1, x2, y1, y2, h, radius, xh, yh)


# Material and Plies properties ( See Documentation )
N = 2
thickness = 0.00027 * np.ones((N, 1))
pho = 3500 * np.ones((N, 1))
angles = np.array([[90],[30]])
TH = sum(thickness) / 2
PPT = np.concatenate(
    (181e9 * np.ones((N, 1)), 10.3e9 * np.ones((N, 1)), 0.3 * np.ones((N, 1)), 7.17e9 * np.ones((N, 1))), axis=1)

pos, Q, Qprime, A, B, D, Klaw = constitutive_law(N, PPT, angles, thickness, TH)
print()

material_param = {'thickness': thickness, 'pho': pho, 'constitutive_law': Klaw, \
                  'Q_matrix': Q, 'angles': angles, 'position': pos, 'Q_prime':Qprime}




script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Example_Cases_Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


""" A test with a Nodal loading in the location (0.7, 0.4)
of  2025 N in the X direction, resulting with a plastic deformation
"""
analysis_type = np.zeros((1,3))
analysis_type[0,0] = 1
analysis_type[0,1] = 2
analysis_type[0,2] = 0
print()
Ngauss=3
print()


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

boundaryconditions = np.array([2, 2, 2, 2])
ENFRCDS = np.zeros((4, 10))
boundary_load = {'NX1': NX1, 'NY1': NY1, 'NX2': NX2, 'NY2': NY2, 'NX3': NX3, \
                 'NY3': NY3, 'NX4': NX4, 'NY4': NY4, 'MY1': MY1, 'MXY1': MXY1, \
                 'MX2': MX2, 'MXY2': MXY2, 'MY3': MY3, 'MXY3': MXY3, 'MX4': MX4, \
                 'MXY4': MXY4, 'boundaryconditions': boundaryconditions, 'ENFRCDS': ENFRCDS}
print()

f = lambda x, y: 0
g = lambda x, y: 0
h = lambda x, y: 0
pointload = -1
NODALLOAD = np.array([2025, 0, 0])
XY = np.zeros((1, 2))
XY[0,0] = 0.7
XY[0,1] = 0.4
load = 3
NODALLOAD1 = np.array([NODALLOAD])
pointload1 = ([])

distload = 10000
ii = 0
while ii < p.shape[0]:
    if m.sqrt(((p[ii, 0] - XY[0, 0]) ** 2) + ((p[ii, 1] - XY[0, 1]) ** 2)) <= distload:
        distload = m.sqrt(((p[ii, 0] - XY[0, 0]) ** 2) + ((p[ii, 1] - XY[0, 1]) ** 2))
        pointload = ii
    ii += 1
pointload1 = np.append(pointload1, pointload)
pointload1 = pointload1.astype(int)

surface_load = {'z': f, 'x': g, 'y': h}
nodal_load = {'coord': pointload1, 'value': NODALLOAD1}



surface_nodal_load = {'surf': surface_load, 'node': nodal_load}

total_loading = {'Bc':boundary_load, 'surf_node':surface_nodal_load}
print()

Nincr = 31
Niter = 100

limit = np.zeros((3, 3))
Xx = 1380e6
Y = 40e6
SLT = 70e6

limit[0, 0] = 1 / Xx
limit[1, 1] = 1 / Y
limit[2, 2] = 1 / SLT

plast_param = {'increments': Nincr, 'iterations': Niter, 'yield_limit': limit}

transient = 0

U, Fb, sxx, syy, sxy, saved_residual, epxx, epyy, epxy, saved_deltaU = \
            FEM(total_loading, p, t, b, Ngauss, box,
                analysis_type, transient, material_param, plast_param)




chosen_step = 28
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
plt.title('Transversal Displacement w (Oz) at step = ' + str(chosen_step))
ax.plot_trisurf(p[:, 0], p[:, 1], t, w, linewidth=0.2, antialiased=True, cmap='jet')
fig.savefig(results_dir + 'CASE 1  Transversal Displacement w (Oz) step = ' + str(chosen_step)+'.png')
plt.show()

print()

fig2 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, u, 500, cmap="jet")
plt.colorbar()
plt.title('Displacement u (Ox) at step = ' + str(chosen_step))
fig2.savefig(results_dir + 'CASE 1  Displacement u (Ox) step = ' + str(chosen_step)+'.png')
plt.show()

fig3 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, v, 500, cmap="jet")
plt.colorbar()
plt.title('Displacement v (Oy) at step = ' + str(chosen_step))
fig3.savefig(results_dir + 'CASE 1  Displacement v (Oy) step = ' + str(chosen_step)+'.png')
plt.show()


fig4 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, thetax, 500, cmap="jet")
plt.colorbar()
plt.title('Rotation on (Ox) at step = ' + str(chosen_step))

fig4.savefig(results_dir + 'CASE 1  Rotation on (Ox) step = ' + str(chosen_step)+'.png')
plt.show()


fig5 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, thetay, 500, cmap="jet")
plt.colorbar()
plt.title('Rotation on (Oy) at step = ' + str(chosen_step))
fig5.savefig(results_dir + 'CASE 1  Rotation on (Oy) step = ' + str(chosen_step)+'.png')
plt.show()


component = 1

if component != 0:
            lay = 1
            chosen_step = 28
            titledef = 'CASE 1  Stress Sxx (Ox)'
            titledef += ' at step '+ str(chosen_step)


            fig9 = plt.figure()
            plt.gca().set_aspect('equal')


            sxx_used = sxx[chosen_step - 1]
            sxx_used = sxx_used[1, 1, :, lay - 1]
            plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=sxx_used, edgecolors='k')

            plt.colorbar()
            plt.title(titledef)
            fig9.savefig(results_dir + titledef + ' .png')
            plt.show()


#Finding the element to plot the stress


x = XY[0,0]
y = XY[0,1]

i = 0
while i < t.shape[0]:
    if x <= np.max(p[t[i,:],0]) and  np.min(p[t[i,:],0]) <= x and  np.min(p[t[i,:],1]) <= y and y <= np.max(p[t[i,:],1]):
        elementu = i
        break
    i += 1

plotted_sxx = np.zeros((1,len(sxx)))
plotted_epxx = np.zeros((1,len(epxx)))
step=0
while step<len(sxx):
    plotted_sxx[0,step] = sxx[step][1,1,elementu,lay - 1]
    ii = 0
    while ii<step:
        plotted_epxx[0,step] += epxx[ii][elementu,0]
        ii+=1

    step+=1

fig6 = plt.figure()
titledef = 'Stress Strain Curve in the applied load region '
plt.plot(plotted_epxx[0], plotted_sxx[0], 'b-')
plt.title(titledef)
fig6.savefig(results_dir + ' CASE 1 '+ titledef + ' .png')
plt.show()

# ----------------------------- END  CASE  1 ----------------------------------------------
#----------------------------------------------------------------------------------------------


#-----------------------    CASE  2  Modal Frequencies --------------------------------
#-------------------------------------------------------------------------------------
#The Geometry
xh=yh=0
x1 = 0
y1 = 0
x2 = 1
y2 = 0.8
h = 0.04
radius = 0.15
xh = 0.2
yh = 0.3
box = np.array([[x1, y1], [x2, y2]])

# Meshing the geometry
p, t, b = mesh(x1, x2, y1, y2, h, radius, xh, yh)


# Material and Plies properties ( See Documentation )
N = 10
thickness = 0.00027 * np.ones((N, 1))
pho = 3500 * np.ones((N, 1))
angles = np.array([[90], [0], [45], [-45], [90], [90], [-45], [45], [0], [90]])
TH = sum(thickness) / 2
PPT = np.concatenate(
    (181e9 * np.ones((N, 1)), 10.3e9 * np.ones((N, 1)), 0.3 * np.ones((N, 1)), 7.17e9 * np.ones((N, 1))), axis=1)

pos, Q, Qprime, A, B, D, Klaw = constitutive_law(N, PPT, angles, thickness, TH)
print()

material_param = {'thickness': thickness, 'pho': pho, 'constitutive_law': Klaw, \
                  'Q_matrix': Q, 'angles': angles, 'position': pos, 'Q_prime':Qprime}




script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Example_Cases_Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


""" A test with a modal analysis
with boundary conditions free-free-free-free
showing thus the rigid and flexible modes
"""
analysis_type = np.zeros((1,3))
analysis_type[0,0] = 3
analysis_type[0,2] = 1
print()
Ngauss=3
print()

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


transient = 0
f = lambda x, y: 0
g = lambda x, y: 0
h = lambda x, y: 0
surface_load = {'z': f, 'x': g, 'y': h}

pointload = ([])
NODALLOAD = np.array([np.array([0, 0, 0])])
NODALLOAD = NODALLOAD[1::, :]
nodal_load = {'coord': pointload, 'value': NODALLOAD}
load = 1
surface_nodal_load = {'surf': surface_load, 'node': nodal_load}



surface_load = {'z': f, 'x': g, 'y': h}
boundaryconditions = np.array([0, 0, 0, 0])
ENFRCDS = np.zeros((4, 10))

boundary_load = {'NX1': NX1, 'NY1': NY1, 'NX2': NX2, 'NY2': NY2, 'NX3': NX3, \
                 'NY3': NY3, 'NX4': NX4, 'NY4': NY4, 'MY1': MY1, 'MXY1': MXY1, \
                 'MX2': MX2, 'MXY2': MXY2, 'MY3': MY3, 'MXY3': MXY3, 'MX4': MX4, \
                 'MXY4': MXY4, 'boundaryconditions': boundaryconditions, 'ENFRCDS': ENFRCDS}
nodal_load = {'coord': pointload, 'value': NODALLOAD}
load = 1
surface_nodal_load = {'surf': surface_load, 'node': nodal_load}

total_loading = {'Bc': boundary_load, 'surf_node': surface_nodal_load}

t1 = time.time()

Kb, Mb, freq, modes, modal_indexes = FEM(total_loading, p, t, b, Ngauss, box,
                                         analysis_type, transient, material_param)

a_file = open(results_dir + 'CASE_2_modal_frequencies.txt', "w")
for row in np.array([freq]):
    np.savetxt(a_file, row)

a_file.close()

t2 = time.time()

mesh_size = p.shape[0]
mode_number = 2

modal_anim = animate_mode(freq, modes, mode_number, modal_indexes, mesh_size, p, t)
writervideo = animation.FFMpegWriter(fps=60)
modal_anim.save(results_dir + "CASE 2 modal_mode_" + str(mode_number) + ".mp4", writer=writervideo)

mode_number = 8

modal_anim = animate_mode(freq, modes, mode_number, modal_indexes, mesh_size, p, t)
writervideo = animation.FFMpegWriter(fps=60)
modal_anim.save(results_dir + "CASE 2 modal_mode_" + str(mode_number) + ".mp4", writer=writervideo)


# ----------------------------- END  CASE  2 ----------------------------------------------
#----------------------------------------------------------------------------------------------


#-----------------    CASE  3 Static Linear Imposed Displacement  ------------------------
#-------------------------------------------------------------------------------------
#The Geometry
xh=yh=0
x1 = 0
y1 = 0
x2 = 1
y2 = 0.8
h = 0.04
radius = 0.15
xh = 0.2
yh = 0.3
box = np.array([[x1, y1], [x2, y2]])

# Meshing the geometry
p, t, b = mesh(x1, x2, y1, y2, h, radius, xh, yh)


# Material and Plies properties ( See Documentation )
N = 10
thickness = 0.00027 * np.ones((N, 1))
pho = 3500 * np.ones((N, 1))
angles = np.array([[90], [0], [45], [-45], [90], [90], [-45], [45], [0], [90]])
TH = sum(thickness) / 2
PPT = np.concatenate(
    (181e9 * np.ones((N, 1)), 10.3e9 * np.ones((N, 1)), 0.3 * np.ones((N, 1)), 7.17e9 * np.ones((N, 1))), axis=1)

pos, Q, Qprime, A, B, D, Klaw = constitutive_law(N, PPT, angles, thickness, TH)
print()

material_param = {'thickness': thickness, 'pho': pho, 'constitutive_law': Klaw, \
                  'Q_matrix': Q, 'angles': angles, 'position': pos, 'Q_prime':Qprime}




script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Example_Cases_Results/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


""" A static Linear Case with imposed displacement
 on the two ends of the plate
"""
analysis_type = np.zeros((1,3))
analysis_type[0,0] = 1
analysis_type[0,1] = 1
analysis_type[0,2] = 1
print()
Ngauss=3
print()


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

boundaryconditions = np.array([4, 0, 4, 0])
ENFRCDS = np.zeros((4, 10))
ENFRCDS[0,0] = -0.00015
ENFRCDS[2,0] = 0.00015
ENFRCDS[0,5] = 1
ENFRCDS[2,5] = 1
ENFRCDS[:, np.arange(0, 5)] = ENFRCDS[:, [0, 1, 2, 4, 3]]
ENFRCDS[:, 3] = -ENFRCDS[:, 3]
ENFRCDS[:, np.arange(2, 5)] = -ENFRCDS[:, np.arange(2, 5)]

ENFRCDS[:, np.arange(5, 10)] = ENFRCDS[:, [5, 6, 7, 9, 8]]

boundary_load = {'NX1': NX1, 'NY1': NY1, 'NX2': NX2, 'NY2': NY2, 'NX3': NX3, \
                 'NY3': NY3, 'NX4': NX4, 'NY4': NY4, 'MY1': MY1, 'MXY1': MXY1, \
                 'MX2': MX2, 'MXY2': MXY2, 'MY3': MY3, 'MXY3': MXY3, 'MX4': MX4, \
                 'MXY4': MXY4, 'boundaryconditions': boundaryconditions, 'ENFRCDS': ENFRCDS}
print()

f = lambda x, y: 0
g = lambda x, y: 0
h = lambda x, y: 0
pointload = -1
NODALLOAD = np.array([0, 0, 0])
XY = np.zeros((1, 2))
load = 1
NODALLOAD1 = np.array([NODALLOAD])
pointload1 = ([])
NODALLOAD1 = NODALLOAD1[1::, :]

surface_load = {'z': f, 'x': g, 'y': h}
nodal_load = {'coord': pointload1, 'value': NODALLOAD1}

surface_nodal_load = {'surf': surface_load, 'node': nodal_load}

transient = 0

total_loading = {'Bc':boundary_load, 'surf_node':surface_nodal_load}
print()

Kb, F, U = FEM(total_loading, p, t, b, Ngauss, box,
               analysis_type, transient, material_param)

thickness = material_param['thickness']
angles = material_param['angles']
Qprime = material_param['Q_prime']
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
fig.savefig(results_dir + 'CASE 3 Transversal Displacement w (Oz) .png')
plt.show()


print()

fig2 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, u, 500, cmap="jet")
plt.colorbar()
plt.title('Displacement u (Ox) ')
fig2.savefig(results_dir + 'CASE 3 Displacement u (Ox) .png')
plt.show()


fig3 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, v, 500, cmap="jet")
plt.colorbar()
plt.title('Displacement v (Oy) ')
fig3.savefig(results_dir + 'CASE 3 Displacement v (Oy) .png')
plt.show()


fig4 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, thetax, 500, cmap="jet")
plt.colorbar()
plt.title('Rotation on (Ox) ')

fig4.savefig(results_dir + 'CASE 3 Rotation on (Ox) .png')
plt.show()


fig5 = plt.figure()
plt.gca().set_aspect('equal')
plt.tricontourf(p[:, 0], p[:, 1], t, thetay, 500, cmap="jet")
plt.colorbar()
plt.title('Rotation on (Oy)')
fig5.savefig(results_dir + 'CASE 3 Rotation on (Oy) .png')
plt.show()


th = 0
SLLt = 1380e6
SLLc = 1430e6
STTt = 40e6
STTc = 240e6
SLT = 70e6

titledef = 'CASE 3 Hoffman criterion'

Hoffman_stress = Hoffman(SLLt, SLLc, STTt, STTc, SLT, p, t, u, v, w, thetax, thetay, th, pos, Qprime, thickness, angles)

if th == 0:

    fig16 = plt.figure()
    plt.gca().set_aspect('equal')

    plt.tripcolor(p[:, 0], p[:, 1], t, facecolors=Hoffman_stress, edgecolors='k')

    plt.colorbar()
    plt.title(titledef)
    fig16.savefig(results_dir + titledef + ' .png')

    plt.show()


# ----------------------------- END  CASE  3 ----------------------------------------------
#----------------------------------------------------------------------------------------------

print()
print('ALL tests finished ')

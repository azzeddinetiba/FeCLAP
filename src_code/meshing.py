# -*-coding:Latin-1 -*

import numpy as np
import distmesh as dm
import matplotlib.pyplot as plt

def mesh(x1, x2, y1, y2, h, radius=0, xh=0, yh=0):


    if radius==0:
     fd = lambda p: dm.drectangle(p, x1, x2, y1, y2)
     fig = plt.figure()
     p, t = dm.distmesh2d(fd, dm.huniform, h, (x1, y1, x2, y2), [(x1, y1), (x1, y2), (x2, y1), (x2, y2)])
    else:
     fd = lambda p: dm.ddiff(dm.drectangle(p, x1, x2, y1, y2),dm.dcircle(p, xh, yh, radius))
     fh = lambda p: h + h*6 * dm.dcircle(p, xh, yh, radius)

     fig = plt.figure()

     p, t = dm.distmesh2d(fd, fh, h, (x1, y1, x2, y2),[(x1, y1), (x1, y2), (x2, y1), (x2, y2)])

    plt.show()



    bbox=(x1, y1, x2, y2)

    ii = 0
    b1=[0]
    while ii<p.shape[0]:
     if abs(p[ii,0] - bbox[0]) < 1e-7:
      b1=np.append(b1, [ii])
     ii+=1
    b1=b1[np.arange(1,b1.shape[0])]

    ii = 0
    b2 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 1] - bbox[1]) < 1e-7:
            b2 = np.append(b2, [ii])
        ii += 1
    b2 = b2[np.arange(1, b2.shape[0])]

    ii = 0
    b3 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 0] - bbox[2]) < 1e-7:
            b3 = np.append(b3, [ii])
        ii += 1
    b3 = b3[np.arange(1, b3.shape[0])]

    ii = 0
    b4 = [0]
    while ii < p.shape[0]:
        if abs(p[ii, 1] - bbox[3]) < 1e-7:
            b4 = np.append(b4, [ii])
        ii += 1
    b4 = b4[np.arange(1, b4.shape[0])]

    b1+=1
    b2+=1
    b3+=1
    b4+=1

    border_size=max(b1.shape[0],b2.shape[0],b3.shape[0],b4.shape[0])
    b=np.zeros((4,border_size))

    b[0,:]=np.concatenate((b1,np.zeros((1,border_size-b1.shape[0]))),axis=None)
    b[1,:]=np.concatenate((b2,np.zeros((1,border_size-b2.shape[0]))),axis=None)
    b[2,:]=np.concatenate((b3,np.zeros((1,border_size-b3.shape[0]))),axis=None)
    b[3,:]=np.concatenate((b4,np.zeros((1,border_size-b4.shape[0]))),axis=None)

    b=b.astype(int)

    #t+=1
    return p, t, b
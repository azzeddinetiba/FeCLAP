# -*-coding:Latin-1 -*
import os
import numpy as np
import math as m
import matplotlib.pyplot as plt
import cv2
import pylab
from scipy import linalg as alg
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy import sparse
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv, jn_zeros
from scipy import sparse as sp


K=sp.eye(5)
F=sp.lil_matrix((5,1))
F+=sp.lil_matrix(np.ones((5,1)))
U = sp.linalg.spsolve(K, F)
print(U.shape)
print(U[0::2])

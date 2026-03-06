# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:48:54 2016

@author: marconi
"""

import numpy as np

def rotaxx(alpha):
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    RX = [ [1,     0,     0],
           [0,  cosa,  sina],
           [0, -sina,  cosa] ]
    return np.array(RX)

def rotaxy(alpha):
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    RY = [ [ cosa, 0, sina],
           [    0, 1,    0],
           [-sina, 0, cosa] ]
    return np.array(RY)
    
def rotaxz(alpha):
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    RZ = [ [ cosa, sina,  0],
           [-sina, cosa,  0],
           [    0,    0,  1] ]
    return np.array(RZ)
    
def eulermat(alpha, beta, gamma):
    # 1. rotation of alpha around z axis
    EM = rotaxz(alpha)
    # 2. rotation of beta around x axis
    EM = np.dot(rotaxx(beta), EM)
    # 3. rotation of gamma around z axis
    EM = np.dot(rotaxz(gamma), EM)
    return EM

def eulermat_inverse(alpha, beta, gamma):
    # 3. rotation of -gamma around z axis
    EMI = rotaxz(-gamma)
    # 2. rotation of -beta around x axis
    EMI = np.dot(rotaxx(-beta), EMI)
    # 1. rotation of -alpha around z axis
    EMI = np.dot(rotaxz(-alpha), EMI)
    return EMI
    
if __name__ == '__main__':
    alpha = np.pi/6
    beta = np.pi/3
    gamma = np.pi/4
    P = (np.array([[1., 0., 3.], [0,0,0], [1,1,1], [2,1,0]])).transpose()
    print(P)
    #print eulermat(np.pi/6, np.pi/3, np.pi/4)
    print(np.dot(eulermat(alpha, beta, gamma), P))
    print('z axis')
    print(np.dot(eulermat_inverse(alpha, beta, gamma), np.array([0., 0., 1.]).transpose()))
   
    
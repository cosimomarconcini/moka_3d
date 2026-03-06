#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:05:10 2018

@author: marconi
"""

import numpy as np

from scipy import special
# import time

from astropy import constants as K


def vel_disk(r, Rd, Mdyn, Rext=5.0, gasSigma=0):
    # r in kpc
    # Rd in kpc
    # Mdyn in Msun

    Vnorm = ((K.G*1e10*K.M_sun/K.kpc)**0.5).to('km/s').value

    y = r/(2.*Rd)
    
    try:
        # start_time = time.time()
        Ay = ( special.iv(0,y)*special.kv(0,y)-special.iv(1,y)*special.kv(1,y) )**0.5   # takes a lot..
      
        # print("rotation curves bessel function time--- %s seconds ---" % (time.time() - start_time))
 
        Ay[y==0] = 0.

      ## special.iv & special.kv sonno funzioni di bessel, non c'e modo di velocizzare questa parte
        
    
    except:
        print(Rd,Mdyn)
        print(special.iv(0,y))
        print(special.kv(0,y))
        print(special.iv(1,y))
        print(special.kv(1,y))

    #V0 = 1e-3*((K.G.value*Mdyn*K.M_sun.value/(Re*K.kpc.value))**0.5) # in km
    V0 = Vnorm*(Mdyn/Rd)**0.5

    B0 = (1.-np.exp(-Rext/Rd)*(1+Rext/Rd))**0.5

    v_circ =  V0/B0*y*Ay*np.sqrt(2.0)

    #    vtot2 = v_circ**2#-(2*(sigma**2)*(r/Re))
    #wneg = np.where(vtot2<0.0)
    #vtot2[wneg] = 0

    #vtot = v_circ


    return v_circ


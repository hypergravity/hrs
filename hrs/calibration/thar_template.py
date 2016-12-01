#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:37:08 2016

@author: cham
"""

#%%
%pylab qt
import os
import numpy as np

from astropy.io import fits
os.chdir('/home/cham/PycharmProjects/hrs/hrs/calibration')

#%%
""" w20160120022t """
wave = np.loadtxt('./w20160120022t_wave.dat')

order = np.loadtxt('./w20160120022t_order.dat')
order = np.repeat(order.reshape(-1, 1), wave.shape[1], axis=1)

thar = fits.open('w20160120022t.fits')[0].data

hl = fits.HDUList(hdus=[fits.PrimaryHDU(),
                        fits.hdu.ImageHDU(wave, name='wave'),
                        fits.hdu.ImageHDU(thar, name='thar'),
                        fits.hdu.ImageHDU(order, name='order')])
                   
hl.writeto('./thar_template/thar_temp_w20160120022t.fits', clobber=True)


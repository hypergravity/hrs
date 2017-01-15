# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:21:30 2017

@author: cham
"""

import os
from astropy.io import fits
import numpy as np

os.chdir('/home/cham/PycharmProjects/hrs/song/calibration')

wave = np.loadtxt('./wave.dat')
thar = np.loadtxt('./thar.dat')
wave = np.flipud(wave)
thar = np.flipud(thar)

order = np.repeat(np.arange(wave.shape[0])[:,None], wave.shape[1], 1)
# np.savetxt('./order.dat', order)


hl = fits.HDUList(hdus=[fits.PrimaryHDU(),
                        fits.hdu.ImageHDU(wave, name='wave'),
                        fits.hdu.ImageHDU(thar, name='thar'),
                        fits.hdu.ImageHDU(order, name='order')])
                   
hl.writeto('./thar_template/thar_template.fits', clobber=True)

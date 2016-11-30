# -*- coding: utf-8 -*-
"""

Author
------
Bo Zhang

Email
-----
bozhang@nao.cas.cn

Created on
----------
- Mon Nov 28 15:00:00 2016

Modifications
-------------
-

Aims
----
- utils for calibration

"""

import numpy as np
from scipy.interpolate import splrep, splev
from astropy.io import fits


def fix_thar_sat_neg(thar1d, arm=20, sat_count=50000):
    # ind_sat_conv
    ind_sat = thar1d >= sat_count
    ind_sat_conv = np.zeros_like(ind_sat)
    x_sat, y_sat = np.where(ind_sat)
    for i in range(len(x_sat)):
        ind_sat_conv[x_sat[i], y_sat[i] - arm:y_sat[i] + arm] = 1
    # ind_neg
    ind_neg = thar1d < 0.
    # combine to ind_bad
    ind_bad = np.logical_or(ind_neg, ind_sat_conv)
    thar1d_fixed = np.where(ind_bad, 0, thar1d)

    return thar1d_fixed


def thar_corr2d(thar1d_fixed, thar_temp, xtrim=(1024, 3072), ytrim=(53, 73),
                x_shiftmax=20, y_shiftmax=5):

    corr2d = np.zeros((1 + 2 * y_shiftmax, 1 + 2 * x_shiftmax))

    xslice_temp = slice(*xtrim)
    yslice_temp = slice(*ytrim)
    print("@Cham: computing 2D cross-correlation...")
    for xofst in range(-x_shiftmax, x_shiftmax + 1, 1):
        for yofst in range(-y_shiftmax, y_shiftmax + 1, 1):
            xslice = slice(xtrim[0] + xofst, xtrim[1] + xofst, 1)
            yslice = slice(ytrim[0] + yofst, ytrim[1] + yofst, 1)
            corr2d[yofst + y_shiftmax, xofst + x_shiftmax] = \
                np.mean(thar1d_fixed[yslice, xslice] *
                        thar_temp[yslice_temp, xslice_temp])
    y_, x_ = np.where(corr2d == np.max(corr2d))
    x_shift = x_shiftmax - x_  # reverse sign
    y_shift = y_shiftmax - y_
    return (x_shift, y_shift), corr2d


def interpolate_wavelength_shift(w, shift, thar_temp, thar1d_fixed):
    xshift, yshift = shift

    xcoord = np.arange(w.shape[1])
    ycoord = np.arange(w.shape[0])

    # for X, no difference, for Y, orders are different
    xcoord_xshift = xcoord + xshift
    ycoord_yshift = np.arange(
        w.shape[0] + thar1d_fixed.shape[0] - thar_temp.shape[0]) + yshift
    # shfit X
    w_x = np.zeros(w.shape)
    for i in range(w.shape[0]):
        s = splrep(xcoord, w[i], k=3, s=0)
        w_x[i] = splev(xcoord_xshift, s)

    # shift Y
    w_x_y = np.zeros((thar1d_fixed.shape[0], w.shape[1]))
    for j in range(w.shape[1]):
        s = splrep(ycoord, w_x[:, j], k=3, s=0)
        w_x_y[:, j] = splev(ycoord_yshift, s)

    return w_x_y


def load_thar_temp(thar_temp_path):
    hl = fits.open(thar_temp_path)
    wave = hl['wave'].data
    thar = hl['thar'].data
    order = hl['order'].data
    return wave, thar, order



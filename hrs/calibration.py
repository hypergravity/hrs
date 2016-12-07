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

import itertools

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit, leastsq


# ############################## #
#      fix thar spectra
# ############################## #

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


# ############################## #
#      2d correlation
# ############################## #

def thar_corr2d(thar1d_fixed, thar_temp,xtrim=(1024, 3072), ytrim=(53, 73),
                x_shiftmax=20, y_shiftmax=5, verbose=False):
    """ determine the shift of *thar1d_fixed* relative to *thar_temp*

    Parameters
    ----------
    thar1d_fixed:
        image whose shift is to be determined
    thar_temp:
        reference image
    xtrim: tuple
        (xmin, xmax)
    ytrim: tuple
        (ymin, ymax)
    x_shiftmax: int
        the max y shift for correlation
    y_shiftmax: int
        the max y shift for correlation

    Returns
    -------
    (x_shift, y_shift), corr2d

    """
    # in case that the input data are int
    thar1d_fixed = np.array(thar1d_fixed).astype(float)
    thar_temp = np.array(thar_temp).astype(float)

    if verbose:
        print("@Cham: computing 2D cross-correlation...")

    # initialize result
    corr2d = np.zeros((1 + 2 * y_shiftmax, 1 + 2 * x_shiftmax))
    # make slice
    xslice_temp = slice(*xtrim)
    yslice_temp = slice(*ytrim)
    # 2D correlation
    for xofst in range(-x_shiftmax, x_shiftmax + 1, 1):
        for yofst in range(-y_shiftmax, y_shiftmax + 1, 1):
            xslice = slice(xtrim[0] + xofst, xtrim[1] + xofst, 1)
            yslice = slice(ytrim[0] + yofst, ytrim[1] + yofst, 1)
            corr2d[yofst + y_shiftmax, xofst + x_shiftmax] = \
                np.mean(thar1d_fixed[yslice, xslice] *
                        thar_temp[yslice_temp, xslice_temp])
    # select maximum value
    y_, x_ = np.where(corr2d == np.max(corr2d))
    x_shift = np.int(x_shiftmax - x_)  # reverse sign
    y_shift = np.int(y_shiftmax - y_)

    return (x_shift, y_shift), corr2d


# ############################## #
#      shift wave & order
# ############################## #

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


def interpolate_order_shift(order_temp, shift, thar1d_fixed):
    xshift, yshift = shift

    # for X, no difference, for Y, orders are different
    ycoord_yshift = np.arange(thar1d_fixed.shape[0]) + \
                    order_temp[0, 0] + yshift
    order_interp = np.repeat(ycoord_yshift.reshape(-1, 1),
                             thar1d_fixed.shape[1], axis=1)

    return order_interp


# ############################## #
#      load temp thar spectra
# ############################## #

def load_thar_temp(thar_temp_path):
    hl = fits.open(thar_temp_path)
    wave = hl['wave'].data
    thar = hl['thar'].data
    order = hl['order'].data
    return wave, thar, order


# ############################## #
#      refine thar positions
# ############################## #

def refine_thar_positions(wave_init, order_init, thar1d_fixed, thar_list,
                          fit_width=5., lc_tol=5., k=3, n_jobs=10, verbose=10):
    print("@Cham: refine ThAr positions ...")

    # refine thar positions for each order
    r = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=1)(
        delayed(refine_thar_positions_order)(
            wave_init[i_order],
            np.arange(wave_init.shape[1]),
            thar1d_fixed[i_order],
            thar_list[(thar_list > np.min(wave_init[i_order]) + 1.) * (
            thar_list < np.max(wave_init[i_order]) - 1.)],
            order_init[i_order, 0],
            fit_width=fit_width, lc_tol=lc_tol, k=k
        ) for i_order in range(wave_init.shape[0]))

    # remove all null values
    null_value = (None, None, None, None, None)
    for i in range(r.count(null_value)):
        r.remove(null_value)
        print(len(r))

    # collect data
    lc_coord = np.array(np.hstack([_[0] for _ in r]))
    lc_order = np.array(np.hstack([_[1] for _ in r]))
    lc_thar = np.array(np.hstack([_[2] for _ in r]))
    popt = np.array(np.vstack([_[3] for _ in r]))
    pcov = np.array(np.vstack([_[4] for _ in r]))

    return lc_coord, lc_order, lc_thar, popt, pcov


def refine_thar_positions_order(this_wave_init, this_xcoord, this_thar,
                                this_thar_list, this_order, fit_width=5.,
                                lc_tol=5., k=3):
    if len(this_thar_list) == 0:
        return None, None, None, None, None

    popt_list = []
    pcov_list = []

    # refine all thar positions in this order
    for i_thar_line, each_thar_line in enumerate(this_thar_list):

        # cut local spectrum
        ind_local = (this_wave_init > each_thar_line - fit_width) * (
            this_wave_init < each_thar_line + fit_width)

        # set bounds
        p0 = (0, 0, 1000., each_thar_line, .05)
        bounds = ((-np.inf, -np.inf, 0., each_thar_line - lc_tol, 0.001),
                  (+np.inf, +np.inf, np.inf, each_thar_line + lc_tol, 2.))

        try:
            popt, pcov = curve_fit(gauss_poly1, this_wave_init[ind_local],
                                   this_thar[ind_local], p0=p0, bounds=bounds)
            pcov = np.diagonal(pcov)
        except RuntimeError:
            popt = np.ones_like(p0) * np.nan
            pcov = np.ones((len(p0),)) * np.nan

        popt_list.append(popt)
        pcov_list.append(pcov)

    # interpolation for X corrdinates
    if np.all(np.diff(this_wave_init) >= 0):
        tck = splrep(this_wave_init, this_xcoord, k=k)
    elif np.all(np.diff(this_wave_init) <= 0):
        tck = splrep(this_wave_init[::-1], this_xcoord[::-1], k=k)
    else:
        raise (ValueError("@Cham: error occurs in interpolation!"))

    # lccov_list = np.array(lccov_list)
    popt_list = np.array(popt_list)
    pcov_list = np.array(pcov_list)

    lc_coord = splev(popt_list[:, 3], tck)
    lc_order = np.ones_like(lc_coord) * this_order

    return lc_coord, lc_order, this_thar_list, popt_list, pcov_list


# ############################## #
#      2D surface fit
# ############################## #

def polyval2d(x, y, coefs, orders=None):
    if orders is None:
        orderx, ordery = coefs.shape
    else:
        orderx, ordery = orders

    ij = itertools.product(range(orderx + 1), range(ordery + 1))
    z = np.zeros_like(x)
    for a, (i, j) in zip(coefs.flatten(), ij):
        if i + j < np.max((orderx, ordery)):
            #        print a,i,j
            z += a * x ** i * y ** j
    return z


def gauss_poly1(x, p0, p1, a, b, c):
    return p0 + p1 * x + a/np.sqrt(2.*np.pi)/c * np.exp(-0.5*((x - b) / c) ** 2.)


def residual_chi2(coefs, x, y, z, w, poly_order):
    return np.nansum(residual(coefs, x, y, z, w, poly_order) ** 2.)


def residual_lar(coefs, x, y, z, w, poly_order):
    fitted = polyval2d(x, y, coefs, poly_order)
    return np.sqrt(np.abs((fitted - z) * w))


def residual(coefs, x, y, z, w, poly_order):
    fitted = polyval2d(x, y, coefs, poly_order)
    return (fitted - z) * w


# ############################## #
#      standardization
# ############################## #

def standardize_inverse(x, xmean, xstd):
    return (x * xstd) + xmean


def standardize(x):
    return (x - np.nanmean(x)) / np.nanstd(x), np.nanmean(x), np.nanstd(x)


# ############################## #
#      fit grating equation
# ############################## #

def fit_grating_equation(lc_coord, lc_order, lc_thar, popt, pcov,
                         poly_order=(3, 5), max_dev_threshold=100, iter=False):
    # pick good thar lines
    try:
        ind_good_thar = np.isfinite(lc_coord) * (popt[:, 4] < 1.0) * (
            pcov[:, 3] < 1.0)
    except:
        ind_good_thar = np.ones_like(lc_coord, dtype=bool)

    lc_coord = lc_coord[ind_good_thar]
    lc_order = lc_order[ind_good_thar]
    lc_thar = lc_thar[ind_good_thar]

    # standardization
    lc_coord_s, lc_coord_mean, lc_coord_std = standardize(lc_coord)
    lc_order_s, lc_order_mean, lc_order_std = standardize(lc_order)
    ml_s, ml_mean, ml_std = standardize(lc_thar * lc_order)
    # scaler
    scaler_coord = lc_coord_mean, lc_coord_std
    scaler_order = lc_order_mean, lc_order_std
    scaler_ml = ml_mean, ml_std

    # weight
    weight = np.ones_like(ml_s)

    # fit surface
    x0 = np.zeros(poly_order)
    x0, ier = leastsq(residual_lar, x0,
                      args=(lc_coord_s, lc_order_s, ml_s, weight, poly_order))

    # iter
    if iter:
        n_loop = 0
        while True:
            n_loop += 1
            x_mini_lsq, ier = leastsq(residual_lar, x0, args=(
                lc_coord_s, lc_order_s, ml_s, weight, poly_order))
            fitted = polyval2d(lc_coord_s, lc_order_s, x_mini_lsq, poly_order)
            fitted_wave = standardize_inverse(fitted, ml_mean, ml_std) / lc_order
            fitted_wave_diff = fitted_wave - lc_thar

            ind_max_dev = np.nanargmax(
                np.abs(fitted_wave_diff - np.nanmedian(fitted_wave_diff)))
            if np.abs(fitted_wave_diff[ind_max_dev] - np.nanmedian(
                    fitted_wave_diff)) > max_dev_threshold:
                weight[ind_max_dev] = 0.
                ml_s[ind_max_dev] = np.nan
                lc_thar[ind_max_dev] = np.nan
                print("@Cham: [n_loop = %s] max_dev = %s" % (
                    n_loop, fitted_wave_diff[ind_max_dev]))
            else:
                if n_loop == 0:
                    print("@Cham: no points cut in iterarions ...")
                break
    else:
        x_mini_lsq, ier = leastsq(residual_lar, x0, args=(
            lc_coord_s, lc_order_s, ml_s, weight, poly_order))
        fitted = polyval2d(lc_coord_s, lc_order_s, x_mini_lsq, poly_order)
        fitted_wave = standardize_inverse(fitted, ml_mean, ml_std) / lc_order
        fitted_wave_diff = fitted_wave - lc_thar

        ind_kick = np.abs(fitted_wave_diff - np.nanmedian(fitted_wave_diff)) > max_dev_threshold
        weight[ind_kick] = 0.
        ml_s[ind_kick] = np.nan
        lc_thar[ind_kick] = np.nan
        x_mini_lsq, ier = leastsq(residual_lar, x0, args=(
            lc_coord_s, lc_order_s, ml_s, weight, poly_order))

    ind_good_thar = np.where(ind_good_thar)[0][np.isfinite(lc_thar)]

    return x_mini_lsq, ind_good_thar, scaler_coord, scaler_order, scaler_ml


def grating_equation_predict(grid_coord, grid_order, x_mini_lsq, poly_order,
                             scaler_coord, scaler_order, scaler_ml):
    sgrid_coord = (grid_coord - scaler_coord[0]) / scaler_coord[1]
    sgrid_order = (grid_order - scaler_order[0]) / scaler_order[1]

    sgrid_fitted = polyval2d(sgrid_coord.flatten(), sgrid_order.flatten(),
                             x_mini_lsq, poly_order)
    sgrid_fitted_wave = (sgrid_fitted * scaler_ml[1] + scaler_ml[0]) / \
                        grid_order.flatten()
    sgrid_fitted_wave = sgrid_fitted_wave.reshape(grid_coord.shape)
    return sgrid_fitted_wave

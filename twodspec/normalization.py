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
- Sat Sep 03 12:00:00 2016

Modifications
-------------
- Sat Sep 03 12:00:00 2016

Aims
----
- normalization

Notes
-----
This is migrated from **SLAM** package

"""

from __future__ import division

import numpy as np
from joblib import Parallel, delayed

from .extern.interpolate import SmoothSpline


def normalize_spectrum(wave, flux, norm_range, dwave,
                       p=(1E-6, 1E-6), q=0.5, ivar=None, eps=1e-10,
                       rsv_frac=1.):
    """ A double smooth normalization of a spectrum

    Converted from Chao Liu's normSpectrum.m
    Updated by Bo Zhang

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux: ndarray (n_pix, )
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    ivar: ndarray (n_pix, ) | None
        ivar array, default is None
    eps: float
        the ivar threshold
    rsv_frac: float
        the fraction of pixels reserved in terms of std. default is 3.

    Returns
    -------
    flux_norm: ndarray
        normalized flux
    flux_cont: ndarray
        continuum flux

    Example
    -------
    >>> flux_norm, flux_cont = normalize_spectrum(
    >>>     wave, flux, (4000., 8000.), 100., p=(1E-8, 1E-7), q=0.5,
    >>>     rsv_frac=2.0)

    """
    if ivar is not None:
        # ivar is set
        ivar = np.where(np.logical_or(wave < norm_range[0],
                                      wave > norm_range[1]), 0, ivar)
        ivar = np.where(ivar <= eps, eps, ivar)
        # mask = ivar <= eps
        var = 1. / ivar
    else:
        # default config is even weight
        var = np.ones_like(flux)

    # wave = wave[~mask]
    # flux = flux[~mask]

    # check q region
    assert 0. < q < 1.

    # n_iter = len(p)
    n_bin = np.int(np.fix(np.diff(norm_range) / dwave) + 1)
    wave1 = norm_range[0]

    # SMOOTH 1
    # print(wave.shape, flux.shape, var.shape)
    if ivar is not None:
        ind_good_init = 1. * (ivar > 0.) * (flux > 0.)
    else:
        ind_good_init = 1. * (flux > 0.)
    ind_good_init = ind_good_init.astype(np.bool)
    # print("@Cham: sum(ind_good_init)", np.sum(ind_good_init))

    flux_smoothed1 = SmoothSpline(wave[ind_good_init], flux[ind_good_init],
                                  p=p[0], var=var[ind_good_init])(wave)
    dflux = flux - flux_smoothed1

    # collecting continuum pixels --> ITERATION 1
    ind_good = np.zeros(wave.shape, dtype=np.bool)
    for i_bin in range(n_bin):
        ind_bin = np.logical_and(wave > wave1 + (i_bin - 0.5) * dwave,
                                 wave <= wave1 + (i_bin + 0.5) * dwave)
        if np.sum(ind_bin > 0):
            # median & sigma
            bin_median = np.median(dflux[ind_bin])
            bin_std = np.median(np.abs(dflux - bin_median))
            # within 1 sigma with q-percentile
            ind_good_ = ind_bin * (
                np.abs(dflux - np.nanpercentile(dflux[ind_bin], q * 100.)) < (
                rsv_frac * bin_std))
            ind_good = np.logical_or(ind_good, ind_good_)

    ind_good = np.logical_and(ind_good, ind_good_init)
    # assert there is continuum pixels
    try:
        assert np.sum(ind_good) > 0
    except AssertionError:
        Warning("@Keenan.normalize_spectrum(): unable to find continuum! ")
        ind_good = np.ones(wave.shape, dtype=np.bool)

    # SMOOTH 2
    # continuum flux
    flux_smoothed2 = SmoothSpline(
        wave[ind_good], flux[ind_good], p=p[1], var=var[ind_good])(wave)
    # normalized flux
    flux_norm = flux / flux_smoothed2

    return flux_norm, flux_smoothed2


def normalize_spectra_block(wave, flux_block, norm_range, dwave,
                            p=(1E-6, 1E-6), q=0.5, ivar_block=None, eps=1e-10,
                            rsv_frac=3., n_jobs=1, verbose=10):
    """ normalize multiple spectra using the same configuration
    This is specially designed for TheKeenan

    Parameters
    ----------
    wave: ndarray (n_pix, )
        wavelegnth array
    flux_block: ndarray (n_obs, n_pix)
        flux array
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    ivar_block: ndarray (n_pix, ) | None
        ivar array, default is None
    eps: float
        the ivar threshold
    rsv_frac: float
        the fraction of pixels reserved in terms of std. default is 3.
    n_jobs: int
        number of processes launched by joblib
    verbose: int / bool
        verbose level

    Returns
    -------
    flux_norm: ndarray
        normalized flux

    """
    if ivar_block is None:
        ivar_block = np.ones_like(flux_block)

    if flux_block.ndim == 1:
        flux_block.reshape(1, -1)
    n_spec = flux_block.shape[0]

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(normalize_spectrum)(
            wave, flux_block[i], norm_range, dwave, p=p, q=q,
            ivar=ivar_block[i], eps=eps, rsv_frac=rsv_frac)
        for i in range(n_spec))

    # unpack results
    flux_norm_block = []
    flux_cont_block = []
    for result in results:
        flux_norm_block.append(result[0])
        flux_cont_block.append(result[1])

    return np.array(flux_norm_block), np.array(flux_cont_block)


def get_stable_pixels(pixel_disp, wave_arm=100, frac=0.20):
    """

    Parameters
    ----------
    pixel_disp: np.ndarray
        dispersion array
    wave_arm: int
        the arm length in terms of pixels
    frac: float
        the reserved fraction, between 0.00 and 1.00

    Returns
    -------
    ind_stable

    """
    ind_stable = np.zeros_like(pixel_disp, dtype=np.bool)

    for i in range(len(ind_stable)):
        edge_l = np.max([i - wave_arm, 0])
        edge_r = np.min([i + wave_arm, len(pixel_disp)])
        if pixel_disp[i] <= \
                np.percentile(pixel_disp[edge_l:edge_r], frac * 100.):
            ind_stable[i] = True

    return ind_stable


# TODO: this is a generalized version
def normalize_spectra(wave_flux_tuple_list, norm_range, dwave,
                      p=(1E-6, 1E-6), q=50, n_jobs=1, verbose=False):
    """ normalize multiple spectra using the same configuration

    Parameters
    ----------
    wave_flux_tuple_list: list[n_obs]
        a list of (wave, flux) tuple
    norm_range: tuple
        a tuple consisting (wave_start, wave_stop)
    dwave: float
        binning width
    p: tuple of 2 ps
        smoothing parameter between 0 and 1:
        0 -> LS-straight line
        1 -> cubic spline interpolant
    q: float in range of [0, 100]
        percentile, between 0 and 1
    n_jobs: int
        number of processes launched by joblib
    verbose: int / bool
        verbose level

    Returns
    -------
    flux_norm: ndarray
        normalized flux

    """
    pass


def test_normaliza_spectra_block():
    import os

    os.chdir('/pool/projects/TheKeenan/data/TheCannonData')

    from TheCannon import apogee
    import matplotlib.pyplot as plt

    tr_ID, wl, tr_flux, tr_ivar = apogee.load_spectra("example_DR10/Data")
    tr_label = apogee.load_labels("example_DR10/reference_labels.csv")

    test_ID = tr_ID
    test_flux = tr_flux
    test_ivar = tr_ivar

    r = normalize_spectra_block(wl, tr_flux, (15200., 16900.), 30., q=0.9,
                                rsv_frac=0.5,
                                p=(1E-10, 1E-10), ivar_block=tr_ivar,
                                n_jobs=10, verbose=10)

    flux_norm, flux_cont = r
    flux_norm = np.array(flux_norm)
    flux_cont = np.array(flux_cont)
    flux_ivar = tr_ivar * flux_cont ** 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(10, 20):
        ofst = i * 0.5
        ax.plot(wl, tr_flux[i] + ofst, 'b')
        ax.plot(wl, flux_cont[i] + ofst, 'r')
    fig.tight_layout()
    fig.savefig(
        '/pool/projects/TheKeenan/data/TheCannonData/test_norm_spec_1.pdf')


if __name__ == '__main__':
    test_normaliza_spectra_block()

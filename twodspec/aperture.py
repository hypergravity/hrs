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
- Fri Nov 25 12:53:24 2016

Modifications
-------------
-

Aims
----
- utils for apertures

"""

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from skimage.filters import gaussian, median
from skimage.morphology import disk
from . import ccdproc_mod as ccdproc
import matplotlib.pyplot as plt
from copy import copy
from .normalization import normalize_spectra_block
from scipy.interpolate import interp1d
from scipy.signal import medfilt2d


# ################################# #
#      find starting points
# ################################# #

def find_local_maximas(pixels, n_adj=2, n_smooth=1, n_sep=1):
    """ find all local maximas

    Parameters
    ----------
    pixels:
        a slice of CCD pixels
    n_adj:
        number of adjacent pixels for maximas
    n_smooth:
        running mean smooth window width
    n_sep:
        the separation since last maxima

    Returns
    -------
    ind of maximas

    """
    pixels = np.array(pixels).flatten()
    n_pix = len(pixels)

    ind_max = np.zeros_like(pixels, dtype=bool)
    # smooth
    for i in np.arange(n_smooth, n_pix - n_smooth):
        pixels[i] = np.mean(np.array(pixels[i - n_smooth:i + 1 + n_smooth]))
    # find maximas
    for i in np.arange(n_adj, n_pix - n_adj):
        ind_max[i] = np.argmax(
            np.array(pixels[i - n_adj:i + 1 + n_adj])) == n_adj and np.sum(
            ind_max[i - n_sep:i]) < 1

    return ind_max


def find_local_minimas(pixels, n_adj=2, n_smooth=1, n_sep=10):
    """ find all local minimas

    Parameters
    ----------
    pixels:
        a slice of CCD pixels
    n_adj:
        number of adjacent pixels for minimas
    n_smooth:
        running mean smooth window width
    n_sep:
        the separation since last minima

    Returns
    -------
    ind of minimas

    """
    pixels = np.array(pixels).flatten()
    n_pix = len(pixels)

    ind_min = np.zeros_like(pixels, dtype=bool)
    # smooth
    for i in np.arange(n_smooth, n_pix - n_smooth):
        pixels[i] = np.mean(np.array(pixels[i - n_smooth:i + 1 + n_smooth]))
    # find minimas
    for i in np.arange(n_adj, n_pix - n_adj):
        ind_min[i] = np.argmin(
            np.array(pixels[i - n_adj:i + 1 + n_adj])) == n_adj and np.sum(
            ind_min[i - n_sep:i]) < 1

    return ind_min


def find_mmax_mmin(im, start_col=2100, n_adj=7, n_smooth=3, n_sep=5,
                   verbose=False):
    """ find reasonable maximas and minimas (cross-kick)

    Parameters
    ----------
    im
    start_col

    Returns
    -------

    """
    start_col_slice = np.sum(im[:, start_col][:, None], axis=1)

    imax = find_local_maximas(start_col_slice, n_adj=n_adj, n_smooth=n_smooth,
                              n_sep=n_sep)
    imin = find_local_minimas(start_col_slice, n_adj=n_adj, n_smooth=n_smooth,
                              n_sep=n_sep)

    # no pixel could be both max&min
    try:
        assert np.sum(imax * imin) == 0
    except:
        raise (ValueError("@TWODSPEC: imax*imin != 0"))

    # cross kick
    smax = np.where(imax)[0]
    smin = np.where(imin)[0]
    smmax = []
    smmin = []
    for i in range(len(smax) - 1):
        imin_this = imin[smax[i]:smax[i + 1]]  # ind array for local mins
        smin_this = np.where(imin_this)[0]  # sub array for local mins
        if len(smin_this) == 1:
            smmin.append(smin_this[0] + smax[i])
        elif len(smin_this) > 1:
            # more than 1 local minimas, find the minimal minima
            smmin.append(smin_this[np.argmin(
                start_col_slice[smax[i]:smax[i + 1]][smin_this])] + smax[i])
    # smin = smmin
    for i in range(len(smmin) - 1):
        imax_this = imax[smmin[i]:smmin[i + 1]]  # ind array for local maxs
        smax_this = np.where(imax_this)[0]  # sub array for local maxs
        if len(smax_this) == 1:
            smmax.append(smax_this[0] + smmin[i])
        elif len(smax_this) > 1:
            # more than 1 local maximas, find the maximal maxima
            smmax.append(smax_this[np.argmax(
                start_col_slice[smmin[i]:smmin[i + 1]][smax_this])] + smmin[i])

    if verbose:
        print("@TWODSPEC: %s possible apertures found!" % len(smmax))

    return smmax, smmin


# ################################# #
#      find apertures
# ################################# #

def find_apertures(im, start_col=2100, max_drift=5, max_apwidth=10,
                   n_pix_goodap=1500, n_adj=7, n_smooth=3, n_sep=5, c=3,
                   verbose=False):
    """ find apertures from image

    Parameters
    ----------
    im:
        image
    start_col:
        start column
    max_drift:
        max_drift in finding an aperture
    max_apwidth:
        local comparison width
    n_pix_goodap:
        a good aperture should be more than this number of pixels

    Returns
    -------
    ymmax_goodap: ndarray
        the y-pixel values for good apertures
    """
    # gaussian smooth if c>0
    if c > 0:
        im = ccdproc.CCDData(gaussian(im, sigma=(c, 0)), unit=im.unit)

    # find max & min
    smmax, smmin = find_mmax_mmin(im, start_col=start_col, n_adj=n_adj,
                                  n_smooth=n_smooth, n_sep=n_sep)

    # initialize results
    ymmax = np.zeros((len(smmax), im.shape[1]))
    ymmax[:, start_col] = smmax
    ymmin = np.zeros((len(smmin), im.shape[1]))
    ymmin[:, start_col] = smmin

    # tracing apertures
    for i_ap in range(ymmax.shape[0]):
        for i_col in np.arange(start_col + 1, im.shape[1]):
            y0 = ymmax[i_ap, i_col - 1]
            y1 = np.argmax(im[np.max((0, y0 - max_apwidth)):np.min(
                (im.shape[0], y0 + 1 + max_apwidth)), i_col].data *
                           im[np.max((0, y0 - max_apwidth)):np.min(
                               (im.shape[0], y0 + 1 + max_apwidth)),
                           i_col - 1].data) + y0 - max_apwidth
            if np.abs(y1 - y0) < max_drift:
                # good ap, continue
                ymmax[i_ap, i_col] = y1
            else:
                break
        for i_col in np.arange(start_col - 1, 0, -1):
            y0 = ymmax[i_ap, i_col + 1]
            y1 = np.argmax(im[np.max((0, y0 - max_apwidth)):np.min(
                (im.shape[0], y0 + 1 + max_apwidth)), i_col].data *
                           im[np.max((0, y0 - max_apwidth)):np.min(
                               (im.shape[0], y0 + 1 + max_apwidth)),
                           i_col + 1].data) + y0 - max_apwidth
            if np.abs(y1 - y0) < max_drift:
                # good ap, continue
                ymmax[i_ap, i_col] = y1
            else:
                break
        if verbose:
            print("@TWODSPEC: tracing aperture [%s] " % i_ap)
    #
    # for i_ap in range(ymmin.shape[0]):
    #     for i_col in np.arange(start_col + 1, im.shape[1]):
    #         y0 = ymmin[i_ap, i_col - 1]
    #         y1 = np.argmin(im[np.max((0, y0 - max_apwidth)):np.min(
    #             (im.shape[0], y0 + 1 + max_apwidth)), i_col].data *
    #                        im[np.max((0, y0 - max_apwidth)):np.min(
    #                            (im.shape[0], y0 + 1 + max_apwidth)),
    #                        i_col - 1].data) + y0 - max_apwidth
    #         if np.abs(y1 - y0) < max_drift:
    #             # good ap, continue
    #             ymmin[i_ap, i_col] = y1
    #         else:
    #             break
    #     for i_col in np.arange(start_col - 1, 0, -1):
    #         y0 = ymmin[i_ap, i_col + 1]
    #         y1 = np.argmin(im[np.max((0, y0 - max_apwidth)):np.min(
    #             (im.shape[0], y0 + 1 + max_apwidth)), i_col].data *
    #                        im[np.max((0, y0 - max_apwidth)):np.min(
    #                            (im.shape[0], y0 + 1 + max_apwidth)),
    #                        i_col + 1].data) + y0 - max_apwidth
    #         if np.abs(y1 - y0) < max_drift:
    #             # good ap, continue
    #             ymmin[i_ap, i_col] = y1
    #         else:
    #             break
    #     print i_ap

    ind_goodap = np.sum(ymmax > 0, axis=1) > n_pix_goodap
    ymmax_goodap = ymmax[ind_goodap, :]
    print("@TWODSPEC: number of good aps (max) = %s " % np.sum(ind_goodap))

    #    ind_goodap = np.sum(ymmax>0, axis=1)>1000
    #    ymmin_goodap = ymmin[ind_goodap, :]
    #    print("@TWODSPEC: gumber of good aps (min)", np.sum(ind_goodap))

    return ymmax_goodap  # , ymmin_goodap


# ################################# #
#      combine & group apertures
# ################################# #

def combine_apertures(imlist, n_jobs=10, find_aps_param_dict=None,
                      verbose=False):
    """ combine apertures found from different FLAT images

    Parameters
    ----------
    imlist: list
        list of FLAT
    find_aps_param_dict:
        the Parameters used in *find_apertures*
    n_jobs:
        n_jobs for finding apertures in parallel

    Returns
    -------
    ap_combine:
        the y-coordinates of good apertures found from FLAT list

    """
    if find_aps_param_dict is None:
        find_aps_param_dict = dict(start_col=2100, max_drift=9, max_apwidth=13,
                                   n_pix_goodap=1000, n_adj=7, n_smooth=3,
                                   n_sep=5, c=3)
    ap_list = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(find_apertures)(im, verbose=verbose, **find_aps_param_dict)
        for im in imlist)

    if verbose:
        print("@TWODSPEC: the numbers of apertures found are: ",
              [ap_.shape[0] for ap_ in ap_list])

    ap_combine = np.vstack(ap_list)
    return ap_combine


def group_apertures(ap_comb, start_col=2100, order_dist=10, verbose=False):
    """ group combined apertures to unique apertures

    Parameters
    ----------
    ap_comb:
        combined apertures
    start_col:
        the starting column number
    order_dist:
        the typical distance between orders

    Returns
    -------
    cheb_coefs: ndarray
        the chebyshev polynomial coefs
    ap_uorder_interp: ndarray
        the y-coordinates of unique orders

    """
    # extract shape
    naps = ap_comb.shape[0]
    npix_disp = ap_comb.shape[1]

    # initialize
    x = np.arange(npix_disp)
    yi_start_col = np.zeros(naps)

    # rough chebyshev fit & interpolate
    for i in range(naps):
        y = ap_comb[i]
        c = np.polynomial.chebyshev.chebfit(x, y, 2, w=y > 0)
        yi_start_col[i] = np.polynomial.chebyshev.chebval(x[start_col], c)
    yi_start_col_sorted = np.sort(yi_start_col)
    yi_start_col_sortarg = np.argsort(yi_start_col)

    # find unique orders
    order_arg_list = []
    c = 0
    for i_uorder in range(naps):
        this_order_yi = yi_start_col_sorted[c]
        this_order_ind = np.abs(
            yi_start_col_sorted - this_order_yi) < order_dist
        this_order_arg = yi_start_col_sortarg[this_order_ind]
        order_arg_list.append(this_order_arg)
        c += len(this_order_arg)
        if c >= len(yi_start_col_sorted):
            break

    # fit & extrapolate
    n_uorder = len(order_arg_list)
    cheb_coefs = []
    ap_uorder_interp = np.zeros((n_uorder, npix_disp))
    for i_uorder in range(n_uorder):
        n_suborder = len(order_arg_list[i_uorder])
        x_data = np.repeat(x[:, None].T, n_suborder, axis=0).flatten()
        y_data = ap_comb[order_arg_list[i_uorder], :].flatten()
        c = np.polynomial.chebyshev.chebfit(x_data, y_data, 2, w=y_data > 0)
        cheb_coefs.append(c)
        ap_uorder_interp[i_uorder] = np.polynomial.chebyshev.chebval(x, c)

    if verbose:
        print("@TWODSPEC: %s unique orders found!" % n_uorder)

    return cheb_coefs, ap_uorder_interp


# ################################# #
#      extract 1d spectra
# ################################# #

def extract_1dspec(im, ap_uorder_interp, ap_width=7, verbose=False):
    naps = ap_uorder_interp.shape[0]
    npix = ap_uorder_interp.shape[1]
    ap_int = ap_uorder_interp.astype(int)

    spec = np.zeros((naps, npix), dtype=float)
    sat_mask = np.zeros((naps, npix), dtype=bool)

    for i in range(naps):
        for j in range(npix):
            yc = ap_int[i, j]
            data = im[yc - ap_width:yc + ap_width + 1, j]
            spec[i, j] = np.sum(data)
            sat_mask[i, j] = np.any(data > 60000)

        if verbose:
            print("@TWODSPEC: extracting 1d spec for order [{0}]".format(i))

    return spec, sat_mask


# ################################# #
#      find ind for all orders
# ################################# #

def find_ind_order(ap_uorder_interp, ccd_shape, edge_len=(10, 20)):
    # find bounds for apertures
    ap_uorder_bounds = np.vstack(
        (ap_uorder_interp[0] - edge_len[0],
         np.diff(ap_uorder_interp, axis=0) * .5 + ap_uorder_interp[:-1],
         ap_uorder_interp[-1] + edge_len[1]))

    # initialize ind_order
    ind_order = np.zeros(ccd_shape, dtype=int)

    # generate coordinates
    x, y = np.arange(ccd_shape[1]), np.arange(ccd_shape[0])
    mx, my = np.meshgrid(x, y)

    # for each apertures
    for i_order in range(ap_uorder_bounds.shape[0] - 1):
        this_order_bound_l = ap_uorder_bounds[i_order, :].reshape(1, -1)
        this_order_bound_u = ap_uorder_bounds[i_order + 1].reshape(1, -1)
        ind_this_order = np.logical_and(my > this_order_bound_l,
                                        my < this_order_bound_u)
        ind_order[ind_this_order] = i_order
        print ("@TWODSPEC: marking pixels of order [%s]" % i_order)
    # for edges
    this_order_bound_l = -1
    this_order_bound_u = ap_uorder_bounds[0].reshape(1, -1)
    ind_this_order = np.logical_and(my > this_order_bound_l,
                                    my < this_order_bound_u)
    ind_order[ind_this_order] = -1

    this_order_bound_l = ap_uorder_bounds[-1].reshape(1, -1)
    this_order_bound_u = ccd_shape[0]
    ind_this_order = np.logical_and(my > this_order_bound_l,
                                    my < this_order_bound_u)
    ind_order[ind_this_order] = -2

    return ind_order


# ############################################### #
#      combine flat according to max counts
# ############################################### #

def combine_flat(flat_list, ap_uorder_interp, sat_count=50000, p=95):
    """ combine flat according to max value under sat_count

    Parameters
    ----------
    flat_list
    ap_uorder_interp
    sat_count
    p

    Returns
    -------

    """
    # find ind_order
    ind_order = find_ind_order(ap_uorder_interp, flat_list[0].shape)

    # unique orders
    uorder = np.unique(ind_order)
    uorder_valid = uorder[uorder >= 0]

    # find max for each order
    im_max = np.vstack(
        [find_each_order_max(im_, ind_order, p) for im_ in flat_list])

    # combine flat
    flat_final = np.zeros_like(flat_list[0])
    flat_origin = np.zeros_like(flat_list[0], dtype=int)
    for uorder_ in uorder:
        # find pixels of this uorder
        ind_this_order = ind_order == uorder_
        # which one should be used for this uorder
        i_im = find_max_under_saturation(im_max[:, uorder_],
                                         sat_count=sat_count)
        # fill the findal image with data from that image
        flat_final = np.where(ind_this_order, flat_list[i_im], flat_final)
        flat_origin[ind_this_order] = i_im
        print("@TWODSPEC: filling data: image [%s] --> uorder [%s]" %
              (i_im, uorder_))

    return flat_final, flat_origin


def find_max_under_saturation(max_vals, sat_count=45000):
    """ find the image with maximum value under saturation count number

    Parameters
    ----------
    max_vals
    sat_count

    Returns
    -------
    the ID of max value under sat_count

    """
    if np.any(max_vals < sat_count):
        asort = np.argsort(max_vals)
        for i in range(1, len(asort)):
            if max_vals[asort[i]] >= sat_count:
                return asort[i - 1]
        return asort[i]
    else:
        return np.argmin(max_vals)


# This is actually not used.
# I've implemented *find_each_order_max* for extracting info
# for combining FLAT
def extract_ap_ridge_counts(im, ap_interp):
    npix = im.shape[1]
    naps = ap_interp.shape[0]

    # float -> int
    ap_int = ap_interp.astype(int)

    spec = np.zeros(ap_int.shape, dtype=float)
    sat_mask = np.zeros(ap_int.shape, dtype=bool)
    for j in range(npix):
        for i in range(naps):
            yc = ap_int[i, j]
            data = im[yc, j]
            spec[i, j] = data.data
    return spec, sat_mask


def find_each_order_max(im, ind_order, p=95):
    """ find the max for each order

    Actually the max value is affected by cosmic rays.
    So, finding the 99/95/90 percentiles could be more useful.


    Parameters
    ----------
    im:
        FLAT image
    ind_order:
        ind of apertures
    p: float
        percentile

    Returns
    -------
    max_list: ndarray


    """
    uorder = np.unique(ind_order)
    uorder_valid = uorder[uorder >= 0]

    max_list = np.zeros_like(uorder_valid)
    for each_uorder in uorder_valid:
        max_list[each_uorder] = np.percentile(
            im[np.where(ind_order == each_uorder)], p)
        print("@TWODSPEC: finding maximum value for order [%s]" % each_uorder)

    return max_list


# ################################# #
#      scatterd light correction
# ################################# #

def substract_scattered_light(im_, ap_uorder_interp, ap_width=10,
                              method='gaussian', method_kwargs=None,
                              shrink=.85):  # 10 should be max

    im = copy(im_)
    assert method in {'gaussian', 'median'}

    ind_inter_order = find_inter_order(im, ap_uorder_interp, ap_width=ap_width)

    # fill edge with inter-order values
    for i_col in range(im.shape[1]):
        ind = np.where(ind_inter_order[:, i_col])[0]
        # print(ind)
        im[0:ind[0], i_col] = np.median(im[ind[0]:ind[0]+3, i_col])
        im[ind[-1]:, i_col] = np.median(im[ind[-1]-3:ind[-1], i_col])
    # plt.figure()
    # plt.imshow(im, interpolation='nearest')

    # initiate scattered light
    im_scattered_light = np.zeros_like(im)

    # interpolation
    x = np.arange(im.shape[0])
    print("@TWODSPEC: interpolating scattered light ...")
    for i_col in range(im.shape[1]):
        w = ind_inter_order[:, i_col]
        y = im[:, i_col]
        I = interp1d(x[w], y[w], 'linear', bounds_error=False, fill_value=0.)
        im_scattered_light[:, i_col] = I(x)

    # smooth
    print("@TWODSPEC: smoothing scattered light ...")
    # print(type(im_scattered_light))
    # print(im_scattered_light)
    # print(method_kwargs)
    if method is 'gaussian':
        im_scattered_light_smooth = gaussian(im_scattered_light * shrink, **method_kwargs)
    elif method is 'median':
        im_scattered_light_smooth = median(im_scattered_light * shrink, **method_kwargs)

    return im - im_scattered_light_smooth, im_scattered_light_smooth


def find_inter_order(im, ap_uorder_interp, ap_width=10):
    ind_inter_order = np.ones_like(im, dtype=bool)

    # generate coordinates
    x, y = np.arange(im.shape[1]), np.arange(im.shape[0])
    mx, my = np.meshgrid(x, y)

    # for each apertures
    for i_order in range(ap_uorder_interp.shape[0] - 1):
        this_order_ridge = ap_uorder_interp[i_order, :].reshape(1, -1)
        ind_this_order = np.logical_and(my > this_order_ridge - ap_width,
                                        my < this_order_ridge + ap_width)
        ind_inter_order = np.where(ind_this_order, False, ind_inter_order)
        print ("@TWODSPEC: *find_inter_order* marking pixels of order [%s]" %
               i_order)
    # # upper edge
    # i_order = 0
    # upper_dist_typical = np.median(np.diff(ap_uorder_interp, axis=0)[0])
    # upper_dist_use = np.int(np.max([upper_dist_typical - 2 * ap_width, 3]))
    # this_order_ridge = ap_uorder_interp[i_order, :].reshape(1, -1)
    # ind_this_order = my < (this_order_ridge - ap_width - upper_dist_use)
    # ind_inter_order = np.where(ind_this_order, False, ind_inter_order)
    #
    # # lower edge
    # i_order = -1
    # lower_dist_typical = np.median(np.diff(ap_uorder_interp, axis=0)[0])
    # lower_dist_use = np.int(np.max([lower_dist_typical - 2 * ap_width, 3]))
    # this_order_ridge = ap_uorder_interp[i_order, :].reshape(1, -1)
    # ind_this_order = my > (this_order_ridge + ap_width + lower_dist_use)
    # ind_inter_order = np.where(ind_this_order, False, ind_inter_order)

    # bounds
    # ind_inter_order[0, :] = True
    # ind_inter_order[-1, :] = True
    # plt.figure()
    # plt.imshow(ind_inter_order, interpolation='nearest')
    # print(ind_inter_order)
    return ind_inter_order


# ################################# #
#      sensitivity variation correction
# ################################# #

def apflatten(flat, ap_uorder_interp, ap_width=(-8, 8), pct=50, **normalization):
    """ IRAF apflatten replication """

    # symmetric case
    if not isinstance(ap_width, tuple):
        ap_width = (-ap_width, ap_width)
    ap_npix = ap_width[1] - ap_width[0] + 1

    # fix apertures
    n_order, disp_npix = ap_uorder_interp.shape

    # get slice along dispersion axis
    order, ofst, xcoord, ycoord = get_aperture_index(ap_uorder_interp,
                                                     ap_width=ap_width)
    # slice 2D FLAT
    flat_slc = flat[ycoord, xcoord]

    # get 1d FLAT by sextract method (simple sum)
    flat1d_simple = sextract_all_aperture(
        flat, ap_uorder_interp, ap_width=ap_width)
    # fit 1d FLAT
    flat1d_cont = normalize_spectra_block(
        xcoord[0], flat1d_simple, norm_range=xcoord[0, [0, -1]], **normalization)[1]
    # tile shape
    flat_slc_cont = np.tile(
        flat1d_cont, (1, ap_npix)).reshape(ap_npix * n_order, -1)
    # caculate profile
    flat1d_profile = np.percentile(flat_slc.data / flat_slc_cont, pct, axis=1)
    # reconstruct sliced 2d FLAT model
    flat_slc_model = flat1d_profile.reshape(-1, 1) * flat_slc_cont
    # normalize 2d FLAT to its model
    flat_slc_norm = flat_slc / flat_slc_model

    # put back data
    flat2d_norm = np.ones_like(flat)
    flat2d_model = np.ones_like(flat)
    flat2d_norm[ycoord, xcoord] = flat_slc_norm
    flat2d_model[ycoord, xcoord] = flat_slc_model

    profile_slc = np.tile(flat1d_profile.reshape(-1, 1), (1, disp_npix))
    profile2d = np.zeros_like(flat)
    profile2d[ycoord, xcoord] = profile_slc

    return dict(
        flat=flat,
        flat_slc=flat_slc,
        flat1d_simple=flat1d_simple,
        flat1d_cont=flat1d_cont,
        flat_slc_cont=flat_slc_cont,
        flat1d_profile=flat1d_profile,
        flat_slc_model=flat_slc_model,
        flat_slc_norm=flat_slc_norm,
        flat2d_norm=flat2d_norm,
        flat2d_model=flat2d_model,
        profile_slc=profile_slc,
        profile2d=profile2d,
    )


def select_a_specific_aperture(order, ofst, x_coord, y_coord, specific_order=0):
    """ select a specific aperture """

    # determine sub-index
    ind = np.where(order == specific_order)[0]

    return order[ind], ofst[ind], x_coord[ind], y_coord[ind]


def sextract_a_specific_aperture(img, ap_uorder_interp, ap_width=(-8, 8), specific_order=0, func=np.sum):
    """ simple extraction for a specific order """

    # symmetric case
    if not isinstance(ap_width, tuple):
        ap_width = (-ap_width, ap_width)

    # fix apertures
    ap_uorder_interp = ap_uorder_interp.astype(int)

    # get slice along dispersion axis
    order, ofst, xcoord, ycoord = get_aperture_index(
        ap_uorder_interp, ap_width=ap_width)
    order, ofst, xcoord, ycoord = select_a_specific_aperture(
        order, ofst, xcoord, ycoord, specific_order=specific_order)

    # extract using a simple sum
    spec1d = func(img[ycoord, xcoord], axis=0)

    return spec1d


def sextract_all_aperture(img, ap_uorder_interp, ap_width=(-8, 8), func=np.sum):
    """ simple extraction for all orders """

    # symmetric case
    if not isinstance(ap_width, tuple):
        ap_width = (-ap_width, ap_width)

    # fix apertures
    ap_uorder_interp = ap_uorder_interp.astype(int)

    # get slice along dispersion axis
    order, ofst, xcoord, ycoord = get_aperture_index(
        ap_uorder_interp, ap_width=ap_width)

    # initiate spec1d
    spec1d = np.zeros(ap_uorder_interp.shape)

    # simple extraction
    for specific_order in range(ap_uorder_interp.shape[0]):
        # get index of this aperture
        order_, ofst_, xcoord_, ycoord_ = select_a_specific_aperture(
            order, ofst, xcoord, ycoord, specific_order=specific_order)
        # extract using a simple sum
        spec1d[specific_order] = func(img[ycoord_, xcoord_], axis=0)

    return spec1d


def get_aperture_index(ap_uorder_interp, ap_width=(-8, 8)):
    """ return X & Y index for all apertures """

    # fix pixel index
    ap_uorder_interp = ap_uorder_interp.astype(int)

    # number of apertures
    n_order = ap_uorder_interp.shape[0]
    npix_apwidth = ap_width[1] - ap_width[0] + 1
    npix_aplength = ap_uorder_interp.shape[1]

    # initiate X & Y
    x_coord = np.arange(npix_aplength, dtype=int).reshape(1, -1).repeat(npix_apwidth*n_order, axis=0)
    y_coord = np.zeros((npix_apwidth*n_order, npix_aplength), dtype=int)

    # determine Y
    for i_order in np.arange(ap_uorder_interp.shape[0], dtype=int):
        row_start = i_order*npix_apwidth
        ofst = np.arange(ap_width[0], ap_width[1]+1, dtype=int).reshape(-1, 1)
        y_coord[row_start:row_start+npix_apwidth] = ap_uorder_interp[i_order] + ofst

    # determine order & ofst
    order = np.arange(n_order, dtype=int).repeat(npix_apwidth)
    ofst = np.arange(ap_width[0], ap_width[1] + 1, dtype=int).reshape(1, -1).repeat(n_order, axis=0).reshape((-1,))

    return order, ofst, x_coord, y_coord


# ################################# #
#      apbackground
# ################################# #

def apbackground(img, ap_uorder_interp, offsetlim=(-5, 5),
                 npix_inter=3, kernel_size=(11, 11)):
    """ determine background/scattered light using inter-aperture pixels """

    img = np.array(img)

    # determine inter-aperture pixels
    ap_lo = np.sum(np.array([3, -3, 1]).reshape(-1, 1) * ap_uorder_interp[0:3], axis=0)
    ap_hi = np.sum(np.array([1, -3, 3]).reshape(-1, 1) * ap_uorder_interp[-3:], axis=0)
    ap_extended = np.vstack((ap_lo, ap_uorder_interp, ap_hi))
    ap_inter = (ap_extended[1:] + ap_extended[:-1]) * .5

    # determine the valley of interp-aperture pixels
    ap_width, sys_offset = apwidth(img, ap_inter, offsetlim=(-8, 8),
                                   ap_npix=npix_inter, method='min')

    # get slice along dispersion axis
    order, ofst, xcoord, ycoord = get_aperture_index(ap_inter, ap_width=ap_width)

    # extract median interp-aperture pixels
    spec1d_inter = sextract_all_aperture(img, ap_inter, ap_width=ap_width, func=np.median)
    spec_slc_tiled = np.tile(spec1d_inter, (1, ap_width[1] - ap_width[0] + 1)
                             ).reshape(-1, ap_inter.shape[1])

    # interpolate median background
    sl = np.zeros_like(img) * np.nan
    for i_col in range(sl.shape[1]):
        # xcoord_ = xcoord[:, i_col]
        ycoord_ = ycoord[:, i_col]
        sl_ = spec_slc_tiled[:, i_col]
        sl_interp = interp1d(ycoord_, sl_, bounds_error=False)(np.arange(sl.shape[0]))

        # fill the ends
        ind_end = np.where(np.isfinite(sl_interp))[0][[0, -1]]
        sl_interp[:ind_end[0]] = sl_interp[ind_end[0]]
        sl_interp[ind_end[-1] + 1:] = sl_interp[ind_end[-1]]
        sl[:, i_col] = sl_interp

    # median filter
    sls = medfilt2d(sl, kernel_size=kernel_size)

    return sls


# ################################# #
#      apwidth
# ################################# #

def apwidth(img, ap_uorder_interp, offsetlim=(-8, 8), ap_npix=10, method='max'):
    """ automatically find ap_width for a given ap_npix """

    img = np.array(img)

    ap_width_max = offsetlim[1] - offsetlim[0] + 1
    assert 1 <= ap_npix <= ap_width_max

    # determine SNR of each offset
    ofst, medsnr_lnsum = apoffset_snr(img, ap_uorder_interp, offsetlim=offsetlim)

    if method is "max":
        istart = np.argmax(medsnr_lnsum)
        sys_offset = ofst[istart]

        i_ap_width_lo, i_ap_width_hi = istart, istart
        i_ap_width_min, i_ap_width_max = 0, len(ofst) - 1

        while (i_ap_width_hi - i_ap_width_lo + 1) < ap_npix:
            if i_ap_width_lo - 1 < i_ap_width_min:
                # towards high end
                i_ap_width_hi += 1
            elif i_ap_width_hi + 1 > i_ap_width_max:
                # towards low end
                i_ap_width_lo -= 1

                # then index is inside
            elif medsnr_lnsum[i_ap_width_hi + 1] >= medsnr_lnsum[
                        i_ap_width_lo - 1]:
                # towards high end
                i_ap_width_hi += 1
            elif medsnr_lnsum[i_ap_width_hi + 1] < medsnr_lnsum[
                        i_ap_width_lo - 1]:
                # towards low end
                i_ap_width_lo -= 1
            else:
                raise ValueError(
                    "@SONG: value error when automatically finding ap_width\n"
                    "{0}".format(medsnr_lnsum.__repr__()))
    elif method is "min":
        istart = np.argmin(medsnr_lnsum)
        sys_offset = ofst[istart]

        i_ap_width_lo, i_ap_width_hi = istart, istart
        i_ap_width_min, i_ap_width_max = 0, len(ofst) - 1

        while (i_ap_width_hi - i_ap_width_lo + 1) < ap_npix:
            if i_ap_width_lo - 1 < i_ap_width_min:
                # towards high end
                i_ap_width_hi += 1
            elif i_ap_width_hi + 1 > i_ap_width_max:
                # towards low end
                i_ap_width_lo -= 1

                # then index is inside
            elif medsnr_lnsum[i_ap_width_hi + 1] <= medsnr_lnsum[
                        i_ap_width_lo - 1]:
                # towards high end
                i_ap_width_hi += 1
            elif medsnr_lnsum[i_ap_width_hi + 1] > medsnr_lnsum[
                        i_ap_width_lo - 1]:
                # towards low end
                i_ap_width_lo -= 1
            else:
                raise ValueError(
                    "@SONG: value error when automatically finding ap_width\n"
                    "{0}".format(medsnr_lnsum.__repr__()))
    else:
        raise ValueError("@SONG: invalid method! method={0}".format(method))

    ap_width = (ofst[i_ap_width_lo], ofst[i_ap_width_hi])

    return ap_width, sys_offset


# ################################# #
#      apoffset
# ################################# #

def apoffset_snr(img, ap_uorder_interp, offsetlim=(-8, 8)):
    """ determine the signal-to-noise ratio for offset """
    img = np.array(img)
    n_offset = offsetlim[1] - offsetlim[0] + 1

    # Poisson error
    var = np.abs(img)
    snr = img/np.sqrt(var)

    # get index
    order, ofst, xcoord, ycoord = get_aperture_index(
        ap_uorder_interp, ap_width=offsetlim)

    # median signal-to-noise ratio for each offset
    medsnr = np.nanmedian(snr[ycoord, xcoord], axis=1)

    medsnr_lnsum = np.zeros((n_offset,), dtype=float)
    for i in np.arange(n_offset, dtype=int):
        medsnr_lnsum[i] = np.nansum(np.log(medsnr[i::n_offset]))

    ofst = np.arange(offsetlim[0], offsetlim[1] + 1, dtype=int)

    return ofst, medsnr_lnsum


# ################################# #
#      apvariance
# ################################# #

def apvariance():
    pass


# ################################# #
#      apsaturation
# ################################# #

def apsaturation(img, adulim=65536., gain=4.0):
    # return the saturation mask of img
    return img > (adulim*gain)


# ################
def get_dispersion_slice_index(ap_uorder_interp, i_order, offset=0):
    indx = np.arange(ap_uorder_interp.shape[1], dtype=int)
    indy = np.array(ap_uorder_interp[i_order]+offset, dtype=int)
    return indy, indx


def get_dispersion_slice(im, ap_uorder_interp, i_order, offset=0):
    indy, indx = get_dispersion_slice_index(
        ap_uorder_interp, i_order, offset=offset)
    return np.array(im)[indy, indx]


def put_dispersion_slice(data, im, ap_uorder_interp, i_order, offset=0):
    # get index
    indy, indx = get_dispersion_slice_index(
        ap_uorder_interp, i_order, offset=offset)

    # put data
    im_ = np.array(copy(im))
    im_[indy, indx] = data
    return im_

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:10:05 2016

@author: cham
"""

import sys

import ccdproc
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from matplotlib import rcParams
from tqdm import trange

from hrs import thar_corr2d

from .utils import scan_files

rcParams.update({"font.size": 20})


def measure_xshift_2dcorr(imfps, bias_temp, flat_temp, xmax=12):
    drifts = np.zeros(len(imfps))
    cfn = sys._getframe().f_code.co_name
    for i in trange(len(imfps), ncols=100, ascii=False, desc="**" + cfn + "**",
                    unit=" file"):
        data = ccdproc.CCDData.read(imfps[i], unit="adu")
        data = ccdproc.subtract_bias(data, bias_temp)
        shift, corr2d = thar_corr2d(data.data, flat_temp.data,
                                    xtrim=(512, 1536), ytrim=(512, 1536),
                                    x_shiftmax=xmax, y_shiftmax=0,
                                    verbose=False)
        drifts[i] = np.int(shift[0])
    return drifts


def check_xdrift(t):
    # check Y shift
    ind_bias = t["IMAGETYP"] == "BIAS"
    print("%s BIAS found" % np.sum(ind_bias))
    ind_flat = t["IMAGETYP"] == "FLAT"
    print("%s FLAT found" % np.sum(ind_flat))
    ind_flati2 = t["IMAGETYP"] == "FLATI2"
    print("%s FLATI2 found" % np.sum(ind_flati2))
    ind_thar = t["IMAGETYP"] == "THAR"
    print("%s THAR found" % np.sum(ind_thar))
    ind_star = t["IMAGETYP"] == "STAR"
    print("%s STAR found" % np.sum(ind_star))

    # Y shift
    sub_flat = np.where(ind_flat)[0]
    sub_flati2 = np.where(ind_flati2)[0]
    sub_star = np.where(ind_star)[0]
    sub_thar = np.where(ind_thar)[0]
    sub_bias = np.where(ind_bias)[0]

    if len(sub_flat) > 0:
        # if there is a flat
        t["astemplate"][sub_flat[0]] = True
        flat_temp = ccdproc.CCDData.read(t["fps"][sub_flat[0]], unit="adu")
    else:
        # no flat, use star instead
        t["astemplate"][sub_star[0]] = True
        flat_temp = ccdproc.CCDData.read(t["fps"][sub_star[0]], unit="adu")

    if len(sub_bias) > 0:
        # if there is a bias
        t["astemplate"][sub_bias[0]] = True
        bias_temp = ccdproc.CCDData.read(t["fps"][sub_bias[0]], unit="adu")
    else:
        # use 300 as bias instead
        bias_temp = ccdproc.CCDData(np.ones((2048, 2048)) * 300, unit="adu")
    flat_temp = ccdproc.subtract_bias(flat_temp, bias_temp)

    # plot figure
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)

    sub = sub_flati2
    print("----- FLATI2 -----")
    if len(sub) > 0:
        drifts = measure_xshift_2dcorr(t["fps"][sub], bias_temp, flat_temp,
                                       xmax=12)
        ax.plot(t["MJD-MID"][sub], drifts, "o", ms=20, mec="r", mfc="None",
                label="FLATI2x%s" % len(sub))
        for i, this_drift in zip(sub, drifts):
            t["xdrift"][i] = this_drift

    sub = sub_flat
    print("------ FLAT ------")
    if len(sub) > 0:
        drifts = measure_xshift_2dcorr(t["fps"][sub], bias_temp, flat_temp,
                                       xmax=12)
        ax.plot(t["MJD-MID"][sub], drifts, "o", ms=20, mec="b", mfc="None",
                label="FLATx%s" % len(sub))
        for i, this_drift in zip(sub, drifts):
            t["xdrift"][i] = this_drift

    sub = sub_thar
    print("------ THAR ------")
    if len(sub) > 0:
        drifts = measure_xshift_2dcorr(t["fps"][sub], bias_temp, flat_temp,
                                       xmax=12)
        ax.plot(t["MJD-MID"][sub], drifts, "^", ms=20, mec="k", mfc="None",
                label="THARx%s" % len(sub))
        for i, this_drift in zip(sub, drifts):
            t["xdrift"][i] = this_drift

    sub = sub_star
    print("------ STAR ------")
    if len(sub) > 0:
        drifts = measure_xshift_2dcorr(t["fps"][sub], bias_temp, flat_temp,
                                       xmax=12)
        ax.plot(t["MJD-MID"][sub], drifts, "s-", c="c", ms=20, mec="c",
                mfc="None", label="STARx%s" % len(sub))
        for i, this_drift in zip(sub, drifts):
            t["xdrift"][i] = this_drift

    # save figure
    plt.grid()
    l = plt.legend()
    l.set_frame_on(False)
    # fig.set_figheight(6)
    # fig.set_figwidth(12)

    plt.xlabel("MJD-MID")
    plt.ylabel("DRIFT/pixel")
    fig.tight_layout()

    ylim = ax.get_ylim()
    plt.ylim(ylim[0] - 2, ylim[1] + 2)
    xlim = ax.get_xlim()
    plt.xlim(xlim[0], xlim[0] + 1.4)

    return t, fig

#
# figpath_pdf = "/hydrogen/song/figs/Xdrift_%s.pdf" % day
# figpath_png = "/hydrogen/song/figs/Xdrift_%s.png" % day
#
# fig.savefig(figpath_pdf)
# print(figpath_pdf)
# fig.savefig(figpath_png)
# print(figpath_png)
#
# # %%
# figpath_fits = "/hydrogen/song/figs/t_%s.fits" % day
#
# t.write(figpath_fits, format="fits", overwrite=True)

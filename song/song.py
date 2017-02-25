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
- Fri Feb 24 16:00:00 2017

Modifications
-------------
-

Aims
----
- Song class

"""

import os
import glob
import sys

import numpy as np
from astropy.io import fits
from tqdm import trange
from joblib import Parallel, delayed, dump, load
from astropy.table import Table
from .utils import scan_files
import astropy.units as u
from .master import combine_image
from skimage.morphology import disk
from twodspec import ccdproc_mod as ccdproc
from twodspec.aperture import (combine_apertures, group_apertures,
                               substract_scattered_light, apflatten)


ALL_IMGTYPE = {
    "BIAS", "FLAT", "FLATI2", "THAR", "THARI2", "STAR", "STARI2", "TEST"}


class Config(object):
    kwds = dict(
        kw_exptime="EXPTIME",
        kw_pregain="PRE_GAIN",
        kw_gain="PRE_GAIN",  # key word for GAIN
        kw_imagetype="IMAGETYP",
        kw_object="OBJECT",
        kw_objname="OBJ-NAME",
        kw_slit="SLIT",
        kw_i2pos="I2POS",
        kw_mjdmid="MJD-MID",
        kw_bvc="BVC",
        kw_filepath="fps",
    )
    read = dict(
        hdu=0,
        unit="adu",
        # kwd={"AUTHOR": "Bo Zhang"}
    )
    rot90 = 1
    gain = dict(
        gain_unit=u.electron / u.adu,
        add_keyword={"AUTHOR": "Bo Zhang",
                     "GAINCORR": True},
    )

    bias_combine = dict(   # ???
        method="average"
    )
    settings = dict(
        rot=90,
        apwidth=15
    )
    apfind_kwds = dict(
        start_col=440,          # starting column
        max_drift=8,            # max drift pixel number
        max_apwidth=10,
        n_pix_goodap=100,
        n_adj=10,
        n_smooth=1,
        n_sep=3,
        c=5                     # gaussian smoothing (x, y)
    )
    scattered_light = dict(
        ap_width=10,
        method='median',
        method_kwargs=dict(selem=disk(5)),
        shrink=1.00,
    )
    normalization = dict(
        dwave=30,
        p=(1E-5, 1E-7),
        q=.5,
        ivar_block=None,
        eps=1e-10,
        rsv_frac=1.,
        n_jobs=10,
        verbose=False,
    )

    def __init__(self):
        pass


class Song(Table):
    """ represent SONG configuration """

    dirpath = ""
    cfg = Config()

    READOUT = None     # master
    BIAS = None        # master
    PATH_BIAS = []

    FLAT = None        # master
    FLAT_BIAS = None
    FLAT_NORM =None
    PATH_FLAT = []

    FLATI2 = None      # master
    FLATI2_BIAS = None
    PATH_FLATI2 = []

    THAR = None        # master
    THAR_BIAS = None
    PATH_THAR = []

    THARI2 = None      # master
    THARI2_BIAS = None
    PATH_THARI2 = []

    STAR = None
    TEST = None

    # aperture contents
    ap_comb = None
    ap_coefs = None
    ap_final = None

    def __init__(self, *args, **kwargs):
        super(Song, self).__init__(*args, **kwargs)
        # add other attributes here
        # balabala

    @staticmethod
    def _init_from_dir(dirpath, verbose=True):
        """ initiate from a directory path

        Parameters
        ----------
        dirpath: string
            a directory path

        Returns
        -------

        """
        assert os.path.exists(dirpath)
        s = Song(scan_files(dirpath, xdriftcol=False, verbose=verbose))
        s.dirpath = dirpath
        return s

    def select(self, colname="default", value="FLAT", method="random",
               n_images=10, return_colname=("fps"), verbose=False):
        """ select some images from list

        Parameters
        ----------
        colname: string
            name of the column that will be matched
        value:
            the specified value
        method: string, {"random", "top", "bottom"}
            the method adopted
        n_images:
            the number of images that will be selected
        return_colname:
            the name(s) of the column that will be returned
        verbose:
            if True, print resutl

        Returns
        -------
        the Song instance

        Examples
        --------
        >>> s.list_image(imagetp="STAR", kwds=["OBJECT"])

        """

        # get the colname of imagetype
        if colname is "default":
            colname = self.cfg.kwds["kw_imagetype"]

        # determine the matched images
        ind_match = np.where(self[colname] == value)[0]
        n_match = len(ind_match)
        if n_match < 1:
            print("@SONG: no images matched!")
            return None

        # determine the number of images to select
        n_images = np.min([n_match, n_images])

        # select according to method
        assert method in {"random", "top", "bottom", "all"}
        if method is "all":
            method = "top"
            n_images = n_match
        if method is "random":
            ind_rand = random_ind(n_match, n_images)
            if return_colname is "ind":
                result = ind_match[ind_rand]
            else:
                result = self[return_colname][ind_match[ind_rand]]
        elif method is "top":
            ind_rand = np.arange(0, n_images, dtype=int)
            if return_colname is "ind":
                result = ind_match[ind_rand]
            else:
                result = self[return_colname][ind_match[ind_rand]]
        elif method is "bottom":
            ind_rand = np.arange(n_match-n_images, n_match, dtype=int)
            if return_colname is "ind":
                result = ind_match[ind_rand]
            else:
                result = self[return_colname][ind_match[ind_rand]]

        # verbose
        if verbose:
            print("@SONG: these are all images selected")
            # here result is a Table
            result.pprint()
            # print("+ ----------------------------------------------")
            # print result
            # for r in result:
            #     print(r)
            # print("+ ----------------------------------------------")

        return result

    def ezselect(self, imgtype="FLAT", method="random", n_images=10,
                 verbose=False):
        return self.select(colname="default", value=imgtype, method=method,
                           n_images=n_images, return_colname="fps",
                           verbose=verbose)

    def list_image(self, imagetp="FLAT", kwds=None, max_print=None):
        list_image(self, imagetp=imagetp, return_col=None, kwds=kwds,
                   max_print=max_print)
        return

    # to add more info in summary
    @property
    def summary(self, colname_imagetype="IMAGETYP", return_data=False):
        """

        Parameters
        ----------
        colname_imagetype: string
            the keyword name for image type, default is "IMAGETYP"
        return_data: bool
            if True, return the summary data

        Returns
        -------
        unique images

        """

        u, uind, uinv, ucts = np.unique(self[colname_imagetype],
                                        return_counts=True, return_index=True,
                                        return_inverse=True)
        # print summary information
        print("=====================================================")
        print("[SUMMARY] {:s}".format(self.dirpath))
        print("=====================================================")
        for i in range(len(u)):
            print("{:10s} {:d}".format(u[i], ucts[i]))
        print("=====================================================")

        # return results
        if return_data:
            return u, uind, uinv, ucts

    def ezmaster(self, imgtype="BIAS", n_images=10, select="random",
                 method="mean", gain_corr=True):
        """

        Parameters
        ----------
        imgtype: string
            {"BIAS", "FLAT", "FLATI2", "THAR", "THARI2",
             "STAR", "STARI2", "TEST"}
        n_images: int
            number of images will be use
        select:
            scheme of selection
        method:
            method of combining
        gain_corr: bool
            if True, do gain correction

        Returns
        -------

        """

        assert select in {"random", "top", "bottom", "all"}
        assert imgtype in ALL_IMGTYPE

        fps = self.ezselect(imgtype=imgtype, method=select,
                                 n_images=n_images)
        if imgtype is "BIAS":
            print("@SONG: setting BIAS & READOUT ...")
            self.BIAS, self.READOUT = combine_image(fps, self.cfg,
                                                    method=method,
                                                    gain_corr=gain_corr)
            self.PATH_BIAS = fps

        else:
            print("@SONG: setting {:s} ...".format(imgtype))
            im = combine_image(fps, self.cfg, method=method,
                               gain_corr=gain_corr)[0]
            self.__setattr__(imgtype, im)
            self.__setattr__("PATH_{0}".format(imgtype), fps)

    def dump(self, fp):
        print("@SONG: save to {0} ...".format(fp))
        dump(self, fp)
        return

    @staticmethod
    def load(fp):
        print("@SONG: load from {0} ...".format(fp))
        return load(fp)
    
    def substract_bias(self, master="FLAT"):
        try:
            assert master in ALL_IMGTYPE
        except AssertionError as ae:
            raise(AssertionError(
                "@SONG: master type not valid!"))

        print("@SONG: substract bias for {0} ...".format(master))
        name = "{0}_BIAS".format(master)
        value = ccdproc.subtract_bias(self.__getattribute__(master), self.BIAS)
        self.__setattr__(name, value)

    def aptrace(self, imgs=None, n_jobs=10, verbose=False):

        # the default image used to trace apertures is FLAT_BIAS
        if imgs is None:
            imgs = self.FLAT_BIAS

        if isinstance(self.FLAT.data, np.ndarray) \
                or isinstance(s.FLAT, np.ndarray):
            imgs = [imgs]

        self.ap_comb = combine_apertures(
            imgs, n_jobs=n_jobs, find_aps_param_dict=self.cfg.apfind_kwds,
            verbose=verbose)
        self.ap_coefs, self.ap_final = group_apertures(
            self.ap_comb, start_col=1024, order_dist=7)

    def substract_scattered_light(self, img):
        img, sl = substract_scattered_light(
            img, self.ap_final, **self.cfg.scattered_light)
        return img, sl

    def apflatten(self, flat, ap_width=(-8, 8), **normalization):
        kwargs = self.cfg.normalization
        for k, v in normalization.items():
            kwargs[k] = v
        self.FLAT_NORM = apflatten(flat, self.ap_final, ap_width=ap_width,
                                   **kwargs)


def random_ind(n, m):
    """ from n choose m randomly """
    return np.argsort(np.random.rand(n,))[:m]


def list_image(t, imagetp="FLAT", return_col=None, kwds=None, max_print=None):
    """ list images with specified IMAGETYP value

    Examples
    --------
    >>> list_image(t2, imagetp="STAR", kwds=["OBJECT"])

    Parameters
    ----------
    t: Table
        catalog of files, generated by *scan_files*
    imagetp: string
        IMAGETYP value
    kwds: list
        optional. additional columns to be displayed
    max_print:
        max line number

    Returns
    -------

    """
    ind_mst = np.where(t["IMAGETYP"] == imagetp)[0]

    if max_print is not None:
        if max_print > len(ind_mst):
            max_print = len(ind_mst)
    else:
        max_print = len(ind_mst)

    print("@SONG: these are all images of type %s" % imagetp)
    print("+ --------------------------------------------------")
    if isinstance(kwds, str):
        kwds = [kwds]
    if kwds is None or kwds == "":
        for i in range(max_print):
            print("+ %04d - %s" % (i, t["fps"][ind_mst[i]]))
    else:
        assert isinstance(kwds, list) or isinstance(kwds, tuple)
        for kwd in kwds:
            try:
                assert kwd in t.colnames
            except AssertionError:
                print("kwd", kwd)
                raise AssertionError()

        for i in range(max_print):
            s = "+ %04d - %s" % (i, t["fps"][ind_mst[i]])
            for kwd in kwds:
                s += "  %s" % t[kwd][ind_mst[i]]
            print(s)
    print("+ --------------------------------------------------")

    if return_col is not None:
        return t[return_col][ind_mst]

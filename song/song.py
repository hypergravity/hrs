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
from astropy.table import Table, Column
from .utils import scan_files
import astropy.units as u
from .master import combine_image, read_image
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
        method="median",
        method_kwargs=dict(selem=disk(5)),
        shrink=1.00,
    )
    normalization = dict(
        dwave=30,
        p=(1E-5, 1E-7),
        q=.5,
        ivar_block=None,
        eps=1e-10,
        rsv_frac=2.,
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

    def select(self, cond_dict=None, method="all", n_select=10,
               returns=("fps"), verbose=False):
        """ select some images from list

        Parameters
        ----------
        cond_dict: dict
            the dict of colname:value pairs
        method: string, {"all", "random", "top", "bottom"}
            the method adopted
        n_images:
            the number of images that will be selected
            if n_images is larger than the number of images matched conditions,
            then n_images is forced to be n_matched
        returns:
            the column name(s) of the column that will be returned
            if returns == 'sub', return the subs of selected images
        verbose:
            if True, print result

        Returns
        -------
        the Song instance

        Examples
        --------
        >>> s.list_image({"IMAGETYP":"STAR"}, returns=["OBJECT"])
        >>> s.select({"IMAGETYP":"THAR", "SLIT":6}, method="all", n_select=200,
        >>>          returns="sub", verbose=False)

        """

        # determine the matched images
        ind_match = np.ones((len(self),), dtype=bool)
        if cond_dict is None or len(cond_dict) < 1:
            print("@SONG: no condition is specified!")
        for k, v in cond_dict.items():
            ind_match = np.logical_and(ind_match, self[k] == v)

        # if no image found
        n_matched = np.sum(ind_match)
        if n_matched < 1:
            print("@SONG: no images matched!")
            return None

        sub_match = np.where(ind_match)[0]
        # determine the number of images to select
        n_return = np.min([n_matched, n_select])

        if verbose:
            print("@SONG: conditions are ", cond_dict)
            print("@SONG: {0} matched & {1} selected & {2} will be returned"
                  "".format(n_matched, n_select, n_return))

        # select according to method
        assert method in {"all", "random", "top", "bottom"}
        sub_rand = np.arange(0, n_matched, dtype=int)
        if method is "all":
            n_return = n_matched
        elif method is "random":
            np.random.shuffle(sub_rand)
            sub_rand = sub_rand[:n_return]
        elif method is "top":
            sub_rand = sub_rand[:n_return]
        elif method is "bottom":
            sub_rand = sub_rand[-n_return:]
        sub_return = sub_match[sub_rand]

        # constructing result to be returned
        if returns is "sub":
            result = sub_return
        else:
            result = self[returns][sub_return]

        # verbose
        if verbose:
            print("@SONG: these are all images selected")
            # here result is a Table
            print(result.__repr__())

        return result

    # #################################### #
    # simplified methods to select subsets
    # #################################### #

    def ezselect_rand(self, cond_dict, n_select=10, returns="sub",
                      verbose=False):
        return self.select(cond_dict=cond_dict, returns=returns,
                           method="random", n_select=n_select, verbose=verbose)

    def ezselect_all(self, cond_dict, n_select=10, returns="sub",
                     verbose=False):
        return self.select(cond_dict=cond_dict, returns=returns,
                           method="all", n_select=n_select, verbose=verbose)

    # TODO: this method will either be updated/deleted
    def list_image(self, imagetp="FLAT", kwds=None, max_print=None):
        list_image(self, imagetp=imagetp, return_col=None, kwds=kwds,
                   max_print=max_print)
        return

    @property
    def describe(self, cfgkeys=("SLIT", "IMAGETYP")):
        """

        Parameters
        ----------
        cfgkeys: tuple
            a pair of keys,

        Returns
        -------

        """
        # initialize result Table
        col0 = [Column(np.unique(self[cfgkeys[0]]), cfgkeys[0])]
        cols = [Column(np.zeros_like(col0[0], dtype=int), key2val) for key2val
                in np.unique(self[cfgkeys[1]])]
        col0.extend(cols)
        result = Table(col0)

        # do statistics & assign to result Table
        unique_result = np.unique(self[cfgkeys], return_counts=True)
        for keyvals_unique, count in zip(*unique_result):
            result[keyvals_unique[1]][
                result[cfgkeys[0]] == keyvals_unique[0]] = count

        return result

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

    # @staticmethod
    def read_image(self, fp, cfg=None, gain_corr=True):
        """ read image """

        # default cfg
        if cfg is None:
            cfg = self.cfg

        # read image
        img = read_image(fp, kwargs_read=cfg.read, gain_corr=gain_corr,
                         kw_gain=cfg.kwds["kw_pregain"],
                         kwargs_gain=cfg.gain, rot90=cfg.rot90)
        return img

    def ezmaster(self, cond_dict, n_images=10, select="random",
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
        try:
            for k in cond_dict.keys():
                assert k in self.colnames
        except:
            print("@SONG: key not found: {0}".format(k))
            raise(ValueError())

        fps = self.ezselect(cond_dict, method=select, n_images=n_images)
        print("fps", fps)

        if cond_dict["IMAGETYP"] is "BIAS":
            print("@SONG: setting BIAS & READOUT ...")
            self.BIAS, self.READOUT = combine_image(fps, self.cfg,
                                                    method=method,
                                                    gain_corr=gain_corr)
            self.PATH_BIAS = fps

        else:
            print("@SONG: setting {:s} ...".format(cond_dict["IMAGETYP"]))
            im = combine_image(fps, self.cfg, method=method,
                               gain_corr=gain_corr)[0]
            self.__setattr__(cond_dict["IMAGETYP"], im)
            self.__setattr__("PATH_{0}".format(cond_dict["IMAGETYP"]), fps)

    def dump(self, fp):
        print("@SONG: save to {0} ...".format(fp))
        dump(self, fp)
        return

    @staticmethod
    def load(fp):
        print("@SONG: load from {0} ...".format(fp))
        return load(fp)
    
    def substract_bias(self, img="FLAT"):
        if img in ALL_IMGTYPE:
            print("@SONG: substract bias for {0} ...".format(img))
            name = "{0}_BIAS".format(img)
            value = ccdproc.subtract_bias(self.__getattribute__(img),
                                          self.BIAS)
            self.__setattr__(name, value)
        else:
            return ccdproc.subtract_bias(img, self.BIAS)

    def aptrace(self, imgs=None, n_jobs=10, verbose=False):

        # the default image used to trace apertures is FLAT_BIAS
        if imgs is None:
            imgs = self.FLAT_BIAS

        if isinstance(self.FLAT.data, np.ndarray) \
                or isinstance(self.FLAT, np.ndarray):
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
        return self.FLAT_NORM


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

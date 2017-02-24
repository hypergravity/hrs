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
- Sun Jan  8 14:00:00 2017

Modifications
-------------
-

Aims
----
- utils for scanning SONG files

"""


import glob
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from tqdm import trange
import ccdproc
from joblib import Parallel, delayed


# scan fits header
def scan_fits_header(fps, verbose=True):
    cfn = sys._getframe().f_code.co_name

    if verbose:
        # use tgdm.trange
        for i in trange(len(fps), ncols=100, ascii=False,
                        desc="**" + cfn + "**", unit=" file"):
            fp = fps[i]

            # read fits
            hl = fits.open(fp)
            hdr_dict = dict(hl[0].header)

            # initiate table
            if i == 0:
                t = Table()
                for k, v in hdr_dict.items():
                    if type(v) is str:
                        t.add_column(
                            Column(data=np.array([]), dtype=object, name=k))
                    else:
                        t.add_column(
                            Column(data=np.array([]), dtype=type(v), name=k))
            # add row
            t.add_row(hdr_dict.values())
    else:
        for i in range(len(fps)):
            fp = fps[i]

            # read fits
            hl = fits.open(fp)
            hdr_dict = dict(hl[0].header)

            # initiate table
            if i == 0:
                t = Table()
                for k, v in hdr_dict.items():
                    if type(v) is str:
                        t.add_column(
                            Column(data=np.array([]), dtype=object, name=k))
                    else:
                        t.add_column(
                            Column(data=np.array([]), dtype=type(v), name=k))
            # add row
            t.add_row(hdr_dict.values())

    # add fps column
    t.add_column(Column(fps, "fps"), 0)

    # fix column dtype
    for i, colname in enumerate(t.colnames):
        if t[colname].dtype == np.dtype("O"):
            newcol = t[colname].astype("str")
            t.remove_column(colname)
            t.add_column(newcol, i)

    return t


def scan_files(dirpath, xdriftcol=True, verbose=True):

    fps_all = glob.glob(dirpath + "/*.fits")
    fps_all.sort()
    t = scan_fits_header(fps_all, verbose=verbose)

    if xdriftcol:
        t.add_column(Column(np.zeros((len(t),), dtype=int) * np.nan, "xdrift"))
        t.add_column(Column(np.zeros((len(t),), dtype=bool), "astemplate"))

    return t


def scan_flux90(t, ind_scan=None, ind_type='bool', pct=90,
                n_jobs=-1, verbose=10, **kwargs):
    """

    Parameters
    ----------
    t: table
        catalog of files
    ind_scan: bool array

    ind_type: string
        {'bool', 'int'}
    pct: float
        default is 90
    n_jobs: int
        number of processes launched
    verbose:
        verbose level
    kwargs:
        ccdproc.CCDData.read() keywords

    Returns
    -------
    f90: array
        the 90th percentile array

    """
    if ind_scan is None:
        ind_scan = np.ones((len(t),), dtype=bool)
        ind_type = 'bool'

    f90_ = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(scan_flux90_)(fp, pct, **kwargs) for fp in t['fps'][ind_scan])

    if ind_type == 'bool':
        f90 = np.array(f90_)*ind_scan
    else:
        f90 = np.zeros((len(t),), dtype=float)
        for i90_, i90 in enumerate(ind_scan):
            f90[i90] = f90_[i90_]

    return f90


def scan_flux90_(fp, pct, **kwargs):
    return np.nanpercentile(ccdproc.CCDData.read(fp, **kwargs), pct)
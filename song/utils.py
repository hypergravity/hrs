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


def unique_type(data):
    """ get the unique types of a list of data """
    tps = [type(_) for _ in data]

    tpunique = []
    while len(tps)> 0:
        tp = tps[0]
        tpunique.append(tp)
        tpcount = tps.count(tp)
        for i in range(tpcount):
            tps.remove(tp)
    return tpunique


# grab fits header
def grab_fits_header(fps, n_jobs=2, verbose=True, *args, **kwargs):
    hs = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(fits.getheader)(fp, *args, **kwargs) for fp in fps)

    # get all keys
    all_keys = list(hs[0].keys())
    for i in range(len(hs)):
        for k in hs[i].keys():
            if k not in all_keys:
                all_keys.append(k)

    # columnize data
    columns = [Column(fps, 'fps')]
    for k in all_keys:
        this_column_data = []
        for h in hs:
            if k in h.keys():
                this_column_data.append(h[k])
            else:
                this_column_data.append(None)
        columns.append(Column(this_column_data, k))

    # change object to str
    columns_updated = []
    for i in range(len(columns)):
        if columns[i].dtype is np.dtype(object):
            # if dtype is object
            uniquetp = unique_type(columns[i].data)
            uniquetp.remove(type(None))
            if len(uniquetp) == 1:
                finaltp = uniquetp[0]
                print(finaltp)
                if finaltp is int:
                    # int, replace None with -9999
                    data = []
                    for v in columns[i].data:
                        if v is None:
                            data.append(-9999)
                        else:
                            data.append(v)
                    columns_updated.append(
                        Column(data, columns[i].name, dtype=finaltp))
                else:
                    # float, etc
                    columns_updated.append(
                        Column(columns[i].data, columns[i].name,
                               dtype=finaltp))
            else:
                # if type mixed, use string
                columns_updated.append(
                    Column(columns[i].data, columns[i].name,
                           dtype=str))
        else:
            # if not object, use it directly
            columns_updated.append(columns[i])

    return Table(columns_updated)


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


def scan_files(dirpath, n_jobs=2, verbose=True, xdriftcol=True):

    fps_all = glob.glob(dirpath + "/*.fits")
    fps_all.sort()
    t = grab_fits_header(fps_all, n_jobs=n_jobs, verbose=verbose)

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
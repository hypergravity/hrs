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


import glob
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from tqdm import trange
import ccdproc
from joblib import Parallel, delayed
from astropy.table import Table
from .utils import scan_files


class Config(object):
    colnames = dict(
        col_imagetype='IMAGETYP',
        col_filepath='fps',
    )

    def __init__(self):
        pass


class Song(Table):
    """ represent SONG configuration """
    dirpath = ""
    cfg = Config()

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
        s = Song(scan_files(dirpath, xdriftcol=False, verbose=verbose))
        s.dirpath = dirpath
        return s

    def select_image(self, colname='default', value='FLAT', method='random',
                     n_images=10, return_colname=('fps'), verbose=False):
        """ select some images from list

        Parameters
        ----------
        colname: string
            name of the column that will be matched
        value:
            the specified value
        method: string, {'random', 'top', 'bottom'}
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
        >>> s.list_image(imagetp='STAR', kwds=['OBJECT'])

        """

        # get the colname of imagetype
        if colname is 'default':
            colname = self.cfg.colnames['col_imagetype']

        # determine the matched images
        ind_match = np.where(self[colname] == value)[0]
        n_match = len(ind_match)
        if n_match < 1:
            print('@SONG: no images matched!')
            return None

        # determine the number of images to select
        n_images = np.min([n_match, n_images])

        # select according to method
        assert method in {'random', 'top', 'bottom', 'all'}
        if method is 'all':
            method = 'top'
            n_images = n_match
        if method is 'random':
            ind_rand = random_ind(n_match, n_images)
            if return_colname is 'ind':
                result = ind_match[ind_rand]
            else:
                result = self[return_colname][ind_match[ind_rand]]
        elif method is 'top':
            ind_rand = np.arange(0, n_images, dtype=int)
            if return_colname is 'ind':
                result = ind_match[ind_rand]
            else:
                result = self[return_colname][ind_match[ind_rand]]
        elif method is 'bottom':
            ind_rand = np.arange(n_match-n_images, n_match, dtype=int)
            if return_colname is 'ind':
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

    def select_image_ez(self, imgtype='FLAT', method='random', n_images=10,
                        verbose=False):
        return self.select_image(colname='default', value=imgtype,
                                 method=method, n_images=n_images,
                                 return_colname='fps', verbose=verbose)

    def list_image(self, imagetp='FLAT', kwds=None, max_print=None):
        list_image(self, imagetp=imagetp, return_col=None, kwds=kwds,
                   max_print=max_print)
        return

    # to add more info in summary
    def summarize(self, colname_imagetype='IMAGETYP', return_data=False):
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
        print("-----------------------------------------------------")
        print("\n[SUMMARY] {:s}".format(self.dirpath))
        print("-----------------------------------------------------")
        for i in range(len(u)):
            print("{:10s} {:d}".format(u[i], ucts[i]))
        print("-----------------------------------------------------")

        # return results
        if return_data:
            return u, uind, uinv, ucts


def random_ind(n, m):
    """ from n choose m randomly """
    return np.argsort(np.random.rand(n,))[:m]


def list_image(t, imagetp='FLAT', return_col=None, kwds=None, max_print=None):
    """ list images with specified IMAGETYP value

    Examples
    --------
    >>> list_image(t2, imagetp='STAR', kwds=['OBJECT'])

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
    ind_mst = np.where(t['IMAGETYP'] == imagetp)[0]

    if max_print is not None:
        if max_print > len(ind_mst):
            max_print = len(ind_mst)
    else:
        max_print = len(ind_mst)

    print("@SONG: these are all images of type %s" % imagetp)
    print("+ --------------------------------------------------")
    if isinstance(kwds, str):
        kwds = [kwds]
    if kwds is None or kwds == '':
        for i in range(max_print):
            print("+ %04d - %s" % (i, t['fps'][ind_mst[i]]))
    else:
        assert isinstance(kwds, list) or isinstance(kwds, tuple)
        for kwd in kwds:
            try:
                assert kwd in t.colnames
            except AssertionError:
                print('kwd', kwd)
                raise AssertionError()

        for i in range(max_print):
            s = "+ %04d - %s" % (i, t['fps'][ind_mst[i]])
            for kwd in kwds:
                s += '  %s' % t[kwd][ind_mst[i]]
            print(s)
    print("+ --------------------------------------------------")

    if return_col is not None:
        return t[return_col][ind_mst]

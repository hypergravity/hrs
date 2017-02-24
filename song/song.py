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
    cfg = Config()

    def __init__(self, *args, **kwargs):
        super(Song, self).__init__(*args, **kwargs)
        # add other attributes here
        # balabala

    @staticmethod
    def _init_from_dir(dirpath):
        """ initiate from a directory path

        Parameters
        ----------
        dirpath: string
            a directory path

        Returns
        -------

        """
        return Song(scan_files(dirpath, xdriftcol=False))

    def select_image(self, colname='default', value='FLAT', method='random',
                     n_images=10, return_colname=('fps'), verbose=False):
        """

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
        assert method in {'random', 'top', 'bottom'}
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


def random_ind(n, m):
    """ from n choose m randomly """
    return np.argsort(np.random.rand(n,))[:m]

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
- Wed Jan   4 14:00:00 2016

Modifications
-------------
-

Aims
----
- SONG echelle spectra extractor

"""

import ccdproc
import numpy as np


def produce_master(t, method="median", imagetp='FLAT', slc=None,
                   ccdread_unit='adu'):
    """

    Parameters
    ----------
    t: astropy.table.Table
        the table of SONG observation
    method:
        the method adopted when combining frames
    imagetp: string
        {'BIAS', 'FLAT', 'FLATI2', 'THAR', 'THARI2'}
    slc: slice
        to denote the fraction of being used
    ccdread_unit: string/unit
        default is 'adu'

    Returns
    -------
    mst: ccdproc.CCDData
        the (combined) master frame

    """
    assert method in {'mean', 'median'}

    # 1. produce ind of master frames

    ind_mst = np.where(t['IMAGETYP'] == imagetp)[0]
    # check for positive number of master frames
    try:
        assert len(ind_mst) > 0
    except AssertionError:
        raise IOError("There is no image of type %s!" % imagetp)

    # in default, combine all masters available
    if slc is not None:
        # combine a fraction of masters available
        assert isinstance(slc, slice)
        ind_mst = ind_mst[slc]
    # check for positive number of master frames
    try:
        assert len(ind_mst) > 0
    except AssertionError:
        raise IOError("There is no image of type %s! (slice is bad)" % imagetp)

    # 2. read master frames
    print("@Cham: trying really hard to produce the final %s frame!" % imagetp)
    mst = ccdproc.combine(','.join(t['fps'][ind_mst]), unit=ccdread_unit,
                          method=method)

    return mst

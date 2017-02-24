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
- Fri Feb 24 21:00:00 2017

Modifications
-------------
-

Aims
----
- utils to process master images

"""

import numpy as np

from twodspec import ccdproc_mod as ccdproc


def combine(fps, s, gain_corr=True, method='average'):
    """ estimate bias & read noise """

    # read bias images
    bias_list = [read_image(fp, s.cfg.read, gain_corr,
                            s.cfg.kwds['kw_pregain'], s.cfg.gain, s.cfg.rot90)
                 for fp in fps]
    bias_array = np.array([im.data.astype(float) for im in bias_list])

    # estimate bias
    if method in {'average', 'mean'}:
        bias = np.mean(bias_array, axis=0)
    elif method is 'median':
        bias = np.median(bias_array, axis=0)
    else:
        raise(ValueError("@SONG: method is not valid!"))

    # estimate readout noise
    readout = np.std(bias_array, axis=0)

    return bias, readout


def read_image(fp, kwargs_read, gain_corr, kw_gain, kwargs_gain, rot90):
    im = ccdproc.CCDData.read(fp, **kwargs_read)
    if gain_corr:
        gain_value = im.header[kw_gain]
        im = ccdproc.gain_correct(im, gain=gain_value, **kwargs_gain)
    if rot90 is not None:
        im.rot90(rot90)
    return im
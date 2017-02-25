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


def combine_image(fps, cfg, method='average', gain_corr=True):
    """ estimate bias & read noise """

    # read bias images
    image_list = [read_image(fp, cfg.read, gain_corr,
                             cfg.kwds['kw_pregain'], cfg.gain, cfg.rot90)
                 for fp in fps]
    image_array = np.array([im.data.astype(float) for im in image_list])
    image_unit = image_list[0].unit

    # estimate bias
    if method in {'average', 'mean'}:
        image_ = ccdproc.CCDData(np.mean(image_array, axis=0), unit=image_unit)
    elif method is 'median':
        image_ = ccdproc.CCDData(np.median(image_array, axis=0), unit=image_unit)
    else:
        raise(ValueError("@SONG: method is not valid!"))

    # estimate readout noise
    image_std = ccdproc.CCDData(np.std(image_array, axis=0), unit=image_unit)

    return image_, image_std


def read_image(fp, kwargs_read, gain_corr, kw_gain, kwargs_gain, rot90):
    im = ccdproc.CCDData.read(fp, **kwargs_read)
    if gain_corr:
        gain_value = im.header[kw_gain]
        im = ccdproc.gain_correct(im, gain=gain_value, **kwargs_gain)
    if rot90 is not None:
        im.rot90(rot90)
    return im
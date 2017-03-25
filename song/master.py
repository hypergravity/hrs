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


# ############################### #
# read_song image
# ############################### #
def read_image(fp, kwargs_read, kwargs_gain, rot90=1):

    # read image
    im = ccdproc.CCDData.read(fp, **kwargs_read)

    # gain correction & add keywords
    gain_value = gain_map(im.meta)
    im = ccdproc.gain_correct(im, gain=gain_value, **kwargs_gain)
    im.gain_value = gain_value
    im.gain_corrected = True

    # rotate image if necessary
    if rot90 is not None:
        im.rot90(rot90)
    im.rot90_n = rot90

    # additional information
    im.fps = fp
    im.config = dict(SLIT=im.meta['SLIT'], IMAGETYP=im.meta["IMAGETYP"])

    return im


def gain_map(meta):
    """ map keywords to true gain values according to the CCD manual """
    # Currently it's just a map to gain=1.1 e-/adu
    return 1.1


def combine_image(fps, cfg, method='average'):
    """ estimate bias & read noise """

    # read bias images
    image_list = [read_image(fp, cfg.read, cfg.gain, rot90=cfg.rot90)
                  for fp in fps]
    image_array = np.array([im.data.astype(float) for im in image_list])
    image_unit = image_list[0].unit
    image_fps = [im.fps for im in image_list]

    # assert the same configuration
    for i in range(len(image_list)):
        assert image_list[0].config == image_list[i].config

    # estimate bias
    if method in {'average', 'mean'}:
        image_comb = ccdproc.CCDData(np.mean(image_array, axis=0), unit=image_unit)
    elif method is 'median':
        image_comb = ccdproc.CCDData(np.median(image_array, axis=0), unit=image_unit)
    else:
        raise(ValueError("@SONG: method is not valid!"))

    # estimate readout noise
    image_std = ccdproc.CCDData(np.std(image_array, axis=0), unit=image_unit)

    # pass meta data
    image_comb.meta = image_list[0].meta
    image_comb.fps = image_fps
    image_std.meta = image_list[0].meta
    image_std.fps = image_fps

    return image_comb, image_std

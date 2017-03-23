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
- to modify ccdproc package

"""

import numpy as np
import ccdproc
from ccdproc import *

__version__ = ccdproc.__version__


class CCDData(ccdproc.CCDData):
    config = None
    gain_value = np.nan
    gain_corrected = False
    fps = None
    # meta already exists in CCDData and could be converted to OrderedDict
    # meta = None

    def __init__(self, *args, **kwargs):
        super(ccdproc.CCDData, self).__init__(*args, **kwargs)

    def rot90(self, k):
        """ Rotate self by 90 degrees in the counter-clockwise direction.

        Parameters
        ----------
        k : integer
            Number of times the array is rotated by 90 degrees.

        """
        self.data = np.rot90(self.data, k)
        return self

    # @property
    # def obscfg(self):
    #     return self._obscfg
    #
    # @obscfg.setter
    # def obscfg(self, value):
    #     self._obscfg = value


def combine(*args, **kwargs):
    return CCDData(ccdproc.combine(*args, **kwargs))



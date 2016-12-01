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
- Wed Nov 23 17:37:24 2016

Modifications
-------------
-

Aims
----
- utils for CCD image processing

"""

from astropy.time import Time
from astropy.io import fits


class Frame(object):
    """ a basic class for constructing an image frame """
    filepath = ''
    type_ = ''

    header = None
    data = None
    read_status = False

    exp_start = Time('2000-01-01T00:00:00.000000000')
    exp_stop = Time('2000-01-01T00:00:01.000000000')
    exp_time = exp_stop - exp_start

    def __init__(self, filepath, type_='sci'):
        """ constructor of Frame instances

        Parameters
        ----------
        filepath: string
            file path
        type_: string
            { 'bias' | 'flat' | 'thar' | 'sci' }

        """
        self.filepath = filepath
        self.type_ = type_
        self.read_status = False

    def read(self):
        """ read data """
        hl = fits.open(self.filepath)
        self.header = hl[0].header
        self.data = hl[0].data
        self.read_status = True
        print('@Cham: data read! [%s]' % self.filepath)

    @property
    def pprint(self):
        print("--------------------------------------------------------------")
        print("    file: %s " % self.filepath)
        print("    type: %s " % self.type_)
        if self.read_status:
            print("  header: ")
            print(self.header.__repr__())
            print("    type: ")
            print(self.data.__repr__())
        print("--------------------------------------------------------------")
        pass


def test():
    import os
    os.chdir('/town/HRS/20161110')
    f = Frame('./20161110001.fits', 'flat')
    f.read()
    # print f.data
    # print f.data.shape
    # print f.header
    f.pprint

if __name__ == '__main__':
    test()

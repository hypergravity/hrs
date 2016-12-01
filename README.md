HRS
===

High Resolution Spectrograph (2.16m) Reduction pipeline.

- Github url: [https://github.com/hypergravity/hrs](https://github.com/hypergravity/hrs)
- PYPI url: [https://pypi.python.org/pypi/hrs](https://pypi.python.org/pypi/hrs)


AUTHOR
======

- Bo Zhang (@NAOC, bozhang@nao.cas.cn)

If you find this code useful in your research, please let me know. Thanks!

Any people are welcome to contribute to this package.


DOCUMENTATION
=============

Will be featured in the near future.


INSTALL
=======
`pip install hrs`


PYTHON VERSIONS AND DEPENDENCIES
================================

- python 2.7.12
- ipython 4.0.0


ISSUES
======
**HRS** is originally designed to work under python2.7, but will support python3.X in the future.


References
==========

Liang Wang has a tutorial for 2.16m HRS data reduction based on **IRAF**. Procedures can be found in 
[http://lwang.info/guides/hrsiraf/](http://lwang.info/guides/hrsiraf/).
This method works well except

1. it takes a long time operating file and clicking during wavelength calibration using *ecidentify* task,
2. in low efficiency area (for HRS it means blue side), orders may not be traced very well and could be wrong.


[Brahm et al. (2016)](https://github.com/rabrahm/ceres) proposed an auto-matic echelle data reduction method called **CERES**.
However, HRS in some aspects is specialized and we may not directly use **CERES**.
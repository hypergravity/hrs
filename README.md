HRS
===

High Resolution Spectrograph (HRS, mounted on Xinglong 2.16m telescope) Reduction pipeline.

Designed for HRS, but also aims to solve the reduction of other Echelle spectrographs.

- Github url: [https://github.com/hypergravity/hrs](https://github.com/hypergravity/hrs)
- PYPI url: [https://pypi.python.org/pypi/hrs](https://pypi.python.org/pypi/hrs)


Author
======

- Bo Zhang (@NAOC, bozhang@nao.cas.cn)

If you find this code useful in your research, please let me know. Thanks!

Any people are welcome to contribute to this package.


Documentation
=============

Will be featured in the near future.


Install
=======
`pip install hrs`


Python version and dependencies
===============================

- python 2.7.12
- ipython 4.0.0


Issues
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
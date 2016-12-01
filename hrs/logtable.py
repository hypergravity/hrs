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
- utils for reading HRS log files

"""

import re
from collections import OrderedDict

import numpy as np
from astropy.table import Table, Column


class LogTable(Table):

    @staticmethod
    def read_log(filepath, verbose=False):
        return log2table(filepath, verbose=verbose)


def log2table(filepath, verbose=False):
    """ process observation log file

    Parameters
    ----------
    filepath: string
        file path of observation log

    """

    # 0. read log data
    f = open(filepath)
    slist = f.readlines()
    sjoin = "".join(slist)
    f.close()

    # 1. extract meta data
    observer = re.search(r"(?<=Observer:)[a-z\s]{1,30}(?=[\t|\r|\n|,])",
                         sjoin, re.I | re.M).group(0).strip()
    institute = re.search(r"(?<=Institute:)[a-z\s]{1,30}(?=[\t|\r|\n|,])",
                          sjoin, re.I | re.M).group(0).strip()
    night_assistant = re.search(
        r"(?<=Night Assistant:)[a-z\s]{1,30}(?=[\t|\r|\n|,])",
        sjoin, re.I | re.M).group(0).strip()
    date = re.search(r"(?<=Date):\d+-\d+-\d+(?=[\t|\r|\n|,])",
                     sjoin, re.I | re.M).group(0).strip()
    seeing = re.search(r"(?<=Seeing):\d+\.\d*\"(?=[\t|\r|\n|,])",
                       sjoin, re.I | re.M).group(0).strip()
    weather = re.search(r"(?<=Weather:).+(?=[\t|\r|\n|,])",
                        sjoin, re.I | re.M).group(0).strip()
    rh = re.search(r"(?<=RH:)[0-9-]+%", sjoin, re.I | re.M).group(0).strip()

    log_meta = OrderedDict(observer=observer, institute=institute,
                           night_assistant=night_assistant, date=date,
                           seeing=seeing, weather=weather, rh=rh)
    if verbose:
        print("@Cham: the meta data")
        print(log_meta)

    # 2.extract log items
    # 2.1 find items | this is where log items start
    for i, s in enumerate(slist):
        if s.find("File") != -1:
            break
    iframe = i + 1
    if verbose:
        print("@Cham: log items start from LINE %s" % iframe)

    # 2.2 initialize table
    colnames = [
        "file", "obj", "exp_start", "exp_time", "RA", "Dec", "epoch", "note"]
    coltypes = [
        "|S50", "|S50", "|S50", "|S50", "|S50", "|S50", "|S50", "|S100"]
    cols_init = [Column(np.array([], dtype=coltype), colname) for
                 colname, coltype in zip(colnames, coltypes)]
    log_tbl = Table(cols_init)
    if verbose:
        log_tbl.pprint()
        print("@Cham: table initialized successfully!")

    # 2.3 iteratively extract log items
    for i in np.arange(iframe, len(slist)):
        # split string & extract infomation
        contents = re.split(r"\s*", slist[i].strip())

        # convert system frames to lower case
        if contents[1].lower() in {"bias", "flat", "thar"}:
            contents[1] = contents[1].lower()

        # add rows in table
        if contents[0].find("-") != -1:
            # if there are multi frames, add multi items
            contents0 = re.split(r"-", contents[0])
            file_start = np.int(contents0[0])
            file_stop = np.int(
                contents0[0][:-len(contents0[1])] + contents0[1])
            for i_file in np.arange(file_start, file_stop + 1):
                new_row = fill_list_to_len(
                    ["%s" % i_file] + contents[1:], len(log_tbl.colnames))
                log_tbl.add_row(new_row)
        else:
            # if there is only one frame, add one item
            new_row = fill_list_to_len(contents, len(log_tbl.colnames))
            log_tbl.add_row(new_row)

    # 3. add meta data to table
    log_tbl.meta = log_meta
    if verbose:
        print("@Cham: log table substantiated successfully!")

    # 4. perfect table
    return perfect_logtable(log_tbl)


def fill_list_to_len(slist, lgoal, tobefilled=""):
    """ fill a list with something to a given length """
    for i in range(lgoal - len(slist)):
        slist.append(tobefilled)
    return slist


def perfect_logtable(log_tbl, fill0=True):
    """ perfect log table """
    # convert exp_time to float
    exptime = log_tbl["exp_time"]
    log_tbl.remove_column("exp_time")
    log_tbl.add_column(exptime.astype(np.float), 3)

    # add filename column
    filename_data = []
    if fill0:
        for _ in log_tbl["file"]:
            m = np.int(_) % 1000
            if m > 99:
                filename_data.append(_[:8] + "00" + _[8:] + ".fits")
            elif m>9:
                filename_data.append(_[:8] + "0" + _[8:] + ".fits")
            else:
                filename_data.append(_ + ".fits")
    else:
        for _ in log_tbl["file"]:
            filename_data.append(_ + ".fits")
    filename = Column(filename_data, "filename")

    log_tbl.add_column(filename)

    return log_tbl


def test():
    return log2table("/town/HRS/20161110test/20161110.txt")


if __name__ == "__main__":
    x = test()
    x.pprint()
    # x.write("/town/HRS/20161110test/20161110.fits")

#!/usr/bin/python3
#-------------------------------------------------------------------------------
    #
    #  Filename       : getBdRate.py
    #  Author         : Huang Leilei
    #  Status         : draft
    #  Created        : 2023-12-27
    #  Description    : calculate B-D rate
    #
#-------------------------------------------------------------------------------
    #

    #
#-------------------------------------------------------------------------------

#*** IMPORT ********************************************************************
import sys
import re
import numpy as np
from getBdRateCore import getBdRateCore


#*** FUNCTION ******************************************************************
# getDat
def getDat(strName, fpt, strTag, datQpThreshold):
    # pop and check type items
    strLineCur = next(fpt).rstrip()
    strTypeAll = re.split("\s{2,}", strLineCur)
    for idx in range(len(strTypeAll)):
        if strTypeAll[idx] != CSTR_TYPE_ALL[idx]:
            assert False, "\n\n[error from {:s}] the {:d}(st/nd/rd/th) type item \"{:s}\" in {:s} is incorrect\n".format(strName, idx, strTypeAll[idx], strTag)

    # pop and check info items
    strLineCur = next(fpt).rstrip()
    strInfoAll = re.split("\s{2,}", strLineCur)
    bolHasElapsed = strInfoAll[-1] == CSTR_INFO_TIME
    if bolHasElapsed:
        strInfoAll = strInfoAll[:-1]
    for idx in range(len(strInfoAll)):
        if strInfoAll[idx] != CSTR_INFO_BFR_ALL[idx % 4]:
            assert False, "\n\n[error from {:s}] the {:d}(st/nd/rd/th) info item \"{:s}\" in {:s} is incorrect\n".format(strName, idx, strInfoAll[idx], strTag)
    if len(strInfoAll) != 4 * len(strTypeAll):
        assert False, "\n\n[error from {:s}] numbers of type and info mismatch!".format(strName)

    # main body
    datFul = {}
    datTime = {}
    for strLineCur in fpt:
        # get info
        if bolHasElapsed:
            [*strDat, strElapsed, strSequence] = strLineCur.split()
        else:
            [*strDat, strSequence] = strLineCur.split()
        [strSequence, strQp] = strSequence.split(sep = "_")
        datQp = int(strQp)
        if datQp < datQpThreshold:
            continue
        if bolHasElapsed:
            if not strSequence in datTime:
                datTime[strSequence] = {}
            datTime[strSequence][datQp] = float(strElapsed)
        # create key seq
        if not strSequence in datFul:
            datFul[strSequence] = {}
        for idx in range(len(strDat) // 4):
            # create key type
            if not CSTR_TYPE_ALL[idx] in datFul[strSequence]:
                datFul[strSequence][CSTR_TYPE_ALL[idx]] = {}

            # create key qp
            if not datQp in datFul[strSequence][CSTR_TYPE_ALL[idx]]:
                datFul[strSequence][CSTR_TYPE_ALL[idx]][datQp] = {}
            else:
                assert False, "\n\n[error from {:s}] sequence {:s} qp {:d} occurs more than once!\n".format(strName, strSequence, datQp)

            # prepare data
            # !!! sure, we can go on with key info, however i don't think it is worth doing
            dat = [float(x) for x in strDat[idx * 4: (idx + 1) * 4]]
            # here 0 for bitrate, 1 for psnr y, 2 for psnr u, 3 for psnr v, 4 for psnr average
            dat.append((dat[1] + dat[2] / datSclCh / datSclCh + dat[3] / datSclCh / datSclCh) / (1 + 1 / datSclCh / datSclCh + 1 / datSclCh / datSclCh))

            # set data
            datFul[strSequence][CSTR_TYPE_ALL[idx % 4]][datQp] = dat
            #print(datFul)

    # close
    fpt.close()

    # return
    return datFul, datTime


#*** PARAMTER ******************************************************************
#--- LOCAL -----------------------------
CSTR_TYPE_ALL     = ("average", "I frame", "P frame", "B frame")
CSTR_INFO_BFR_ALL = ("bitrate(kb/s)", "psnr(Y)", "psnr(U)", "psnr(V)")
CSTR_INFO_AFT_ALL = ("bdrate(Y)", "bdrate(U)", "bdrate(V)", "bdrate(average)")
CSTR_INFO_TIME    = "elapsed(s)"
CSTR_USAGE        = "\n[information from {:s}] usage: getBdRate.py anchor.log result.log [YUV420|YUV444 [<QP threshold>]] > bdRate.log\n".format(sys.argv[0])


#*** MAIN **********************************************************************
#--- PREV ------------------------------
# open anchor
try:
    fptAnchor = open(sys.argv[1], "r")
except:
    assert False, "\n\n[error from {:s}] cannot open the anchor file {:s}!".format(sys.argv[0], sys.argv[1]) + CSTR_USAGE

# open testor
try:
    fptResult = open(sys.argv[2], "r")
except:
    assert False, "\n\n[error from {:s}] cannot open the result file {:s}!".format(sys.argv[0], sys.argv[2]) + CSTR_USAGE

# get format
if len(sys.argv) <= 3 or sys.argv[3] == "YUV420":
    datSclCh = 2
elif sys.argv[3] == "YUV444":
    datSclCh = 1
else:
    assert False, "\n\n[error from {:s}] unknown format \"{:s}\"\n".format(sys.argv[0], sys.argv[3]) + CSTR_USAGE

# get format
if len(sys.argv) <= 4:
    datQpThreshold = 0
elif sys.argv[4].isdigit():
    datQpThreshold = int(sys.argv[4])
else:
    assert False, "\n\n[error from {:s}] unknown format \"{:s}\"\n".format(sys.argv[0], sys.argv[4]) + CSTR_USAGE
print("QP less than {:d} is skipped\n".format(datQpThreshold))

# check redundant parameter
if len(sys.argv) > 5:
    assert False, "\n\n[error from {:s}] unknown parameter \"{:s}\"\n".format(sys.argv[0], sys.argv[5:]) + CSTR_USAGE

# process anchor
datAnchor, datTimeAnchor = getDat(sys.argv[0], fptAnchor, "anchor", datQpThreshold)

# process testor
datResult, datTimeResult = getDat(sys.argv[0], fptResult, "result", datQpThreshold)

# Compute one time-saving value per sequence over all QPs present in both runs.
timeSavings = {}
for strSequence in datTimeAnchor:
    if strSequence in datTimeResult:
        datQps = set(datTimeAnchor[strSequence]) & set(datTimeResult[strSequence])
        datTimeAnchorSum = sum(datTimeAnchor[strSequence][datQp] for datQp in datQps)
        datTimeResultSum = sum(datTimeResult[strSequence][datQp] for datQp in datQps)
        if datTimeAnchorSum:
            timeSavings[strSequence] = (datTimeAnchorSum - datTimeResultSum) / datTimeAnchorSum * 100.0


#--- CORE ------------------------------
# head
print("{:<57s} {:<57s} {:<57s} {:s}".format(*CSTR_TYPE_ALL))
for x in range(4):
    print("{:<12s} {:<12s} {:<12s} {:<18s} ".format(*CSTR_INFO_AFT_ALL), end = "")
if timeSavings:
    print("{:<18s} {:s}".format("time saving(%)", "sequence"))
else:
    print("{:s}".format("sequence"))

# body
# for sequence
datBdRtAll = {}
for strSequence in datAnchor:
    if strSequence in datResult:

        # for type
        for strTyp in CSTR_TYPE_ALL:
            datBdRt = []
            if strTyp in datAnchor[strSequence] and strTyp in datResult[strSequence]:

                # for info
                for idxInfoAft in range(len(CSTR_INFO_AFT_ALL)):
                    strInfoAft = CSTR_INFO_AFT_ALL[idxInfoAft]
                    if strInfoAft != "bdrate(average)":    # if enabled, average is taken after  function getBdRateCore
                    #if True:                              # if enabled, average is taken before function getBdRateCore

                        # for qp
                        datBtRtAnchor = []
                        datPsnrAnchor = []
                        datBtRtResult = []
                        datPsnrResult = []
                        # !!! sure, the qp of anchor and result do not have to be identical
                        for datQp in datAnchor[strSequence][strTyp]:
                            # !!!                                                 1 + idxInfoAft here indicates
                            # !!! there is a fixed position constraints between CSTR_INFO_BFR_ALL and CSTR_INFO_AFT_ALL
                            datBtRtAnchor.append(datAnchor[strSequence][strTyp][datQp][0             ])
                            datPsnrAnchor.append(datAnchor[strSequence][strTyp][datQp][1 + idxInfoAft])
                        for datQp in datResult[strSequence][strTyp]:
                            datBtRtResult.append(datResult[strSequence][strTyp][datQp][0             ])
                            datPsnrResult.append(datResult[strSequence][strTyp][datQp][1 + idxInfoAft])

                        # calculate bd rate
                        if any(datBtRtResult) and any(datPsnrResult) and any(datBtRtAnchor) and any(datPsnrAnchor):
                            datBtRtAnchor = np.array(datBtRtAnchor)
                            datPsnrAnchor = np.array(datPsnrAnchor)
                            datBtRtResult = np.array(datBtRtResult)
                            datPsnrResult = np.array(datPsnrResult)
                            datBdRt.append(getBdRateCore(datBtRtAnchor, datPsnrAnchor, datBtRtResult, datPsnrResult))
                        else:
                            datBdRt.append(0)
                    else:
                        datBdRt.append((datBdRt[0] + datBdRt[1] / datSclCh / datSclCh + datBdRt[2] / datSclCh / datSclCh) / (1 + 1 / datSclCh / datSclCh + 1 / datSclCh / datSclCh))
            else:
                datBdRt = (0, 0, 0, 0)

            # collect
            if not strTyp in datBdRtAll:
                datBdRtAll[strTyp] = {}
            for idxInfoAft in range(len(CSTR_INFO_AFT_ALL)):
                strInfoAft = CSTR_INFO_AFT_ALL[idxInfoAft]
                if not strInfoAft in datBdRtAll[strTyp]:
                    datBdRtAll[strTyp][strInfoAft] = []
                datBdRtAll[strTyp][strInfoAft].append(datBdRt[idxInfoAft])

            # dump datBdRt
            print("{:<12.3f} {:<12.3f} {:<12.3f} {:<18.3f} ".format(*datBdRt), end = "")

        # dump strSequence
        if timeSavings:
            print("{:<18.3f} {:s}".format(timeSavings.get(strSequence, float("nan")), strSequence))
        else:
            print(strSequence)

# dump datBdRtStat
print("")
for strStat in ("min", "AVE", "max"):
    for idxLine in range(2):
        for strTyp in CSTR_TYPE_ALL:
            if (idxLine == 0):
                if (strTyp == CSTR_TYPE_ALL[-1]):
                    print("{:<12s} {:<12s} {:<12s} {:s}"    .format(*(x.replace("bdrate", strStat) for x in CSTR_INFO_AFT_ALL)), end = "")
                else:
                    print("{:<12s} {:<12s} {:<12s} {:<18s} ".format(*(x.replace("bdrate", strStat) for x in CSTR_INFO_AFT_ALL)), end = "")
            else:
                func = {"min": np.min, "AVE": np.mean, "max": np.max}[strStat]
                if (strTyp == CSTR_TYPE_ALL[-1]):
                    print("{:<12.3f} {:<12.3f} {:<12.3f} {:.3f}"    .format(*(func(datBdRtAll[strTyp][strInfoAft]) for strInfoAft in CSTR_INFO_AFT_ALL)), end = "")
                else:
                    print("{:<12.3f} {:<12.3f} {:<12.3f} {:<18.3f} ".format(*(func(datBdRtAll[strTyp][strInfoAft]) for strInfoAft in CSTR_INFO_AFT_ALL)), end = "")
        if timeSavings:
            if idxLine == 0:
                print(" {:<18s}".format(strStat + "(time saving)"), end = "")
            else:
                func = {"min": np.min, "AVE": np.mean, "max": np.max}[strStat]
                print(" {:<18.3f}".format(func(list(timeSavings.values()))), end = "")
        print("")

#!/usr/bin/python3
#-------------------------------------------------------------------------------
    #
    #  Filename       : getBdRateCore.py
    #  Author         : Liu Chang
    #  Status         : draft
    #  Created        : 2023-12-27
    #  Description    : calculate B-D rate (core)
    #
#-------------------------------------------------------------------------------

#*** IMPORT ********************************************************************
import numpy as np


#--- BEGIN of unmaintained codes ---
#*** CLASS *********************************************************************
# PCHIP
class PCHIP:
    def __init__(self, x, y):
        assert len(np.unique(x)) == len(x)
        order = np.argsort(x)
        self.xi, self.yi = x[order], y[order]

        h = np.diff(self.xi)
        delta = np.diff(self.yi) / h

        self.d = pchip_slopes(h, delta)
        self.c = (3*delta - 2*self.d[:-1] - self.d[1:]) / h
        self.b = (self.d[:-1] - 2*delta + self.d[1:]) / h**2

        """
        The piecewise function is like p(x) = y_k + s*d_k + s*s*c_k + s*s*s*b_k
        where s = x - xi_k, k is the interval includeing x.
        So the original function of p(x) is P(x) = xi_k*y_k + s*y_k + 1/2*s*s*d_k + 1/3*s*s*s*c_k + 1/4*s*s*s*s*b_k + C.
        """
        self.interval_int_coeff = []
        self.interval_int = np.zeros(len(x)-1)
        for i in range(len(x)-1):
            self.interval_int_coeff.append(np.polyint([self.b[i], self.c[i], self.d[i], self.yi[i]]))
            self.interval_int[i] = np.polyval(self.interval_int_coeff[-1], h[i]) - np.polyval(self.interval_int_coeff[-1], 0)

    def _integral(self, lower, upper):
        assert lower <= upper
        if lower < np.min(self.xi):
            lower = np.min(self.xi)
            #print('Warning: The lower bound is less than the interval and clipped!')
        elif lower > np.max(self.xi):
            #print('Warning: The lower bound is greater than the interval!')
            return 0
        if upper > np.max(self.xi):
            upper = np.max(self.xi)
            #print('Warning: The upper bound is greater than the interval and clipped!')
        elif upper < np.min(self.xi):
            #print('Warning: The lower bound is less than the interval!')
            return 0
        left = np.arange(len(self.xi))[self.xi - lower > -1e-6][0]
        right = np.arange(len(self.xi))[self.xi - upper < 1e-6][-1]
        inte = np.sum(self.interval_int[left:right])
        if self.xi[left] - lower > 1e-6:
            inte += (np.polyval(self.interval_int_coeff[left-1], self.xi[left]-self.xi[left-1]) - np.polyval(self.interval_int_coeff[left-1], lower-self.xi[left-1]))
        if self.xi[right] - upper < -1e-6:
            inte += (np.polyval(self.interval_int_coeff[right], upper-self.xi[right]) - np.polyval(self.interval_int_coeff[right], 0))
        return inte

    def integral(self, lower, upper):
        if lower > upper:
            return -self._integral(upper, lower)
        else:
            return self._integral(lower, upper)


#*** FUNCTION ******************************************************************
# pchip_slopes
def pchip_slopes(h, delta):
    d = np.zeros(len(h) + 1)
    k = np.argwhere(np.sign(delta[:-1]) * np.sign(delta[1:]) > 0).reshape(-1) + 1
    w1 = 2*h[k] + h[k-1]
    w2 = h[k] + 2*h[k-1]
    d[k] = (w1 + w2) / (w1 / delta[k-1] + w2 / delta[k])
    d[0] = pchip_end(h[0], h[1], delta[0], delta[1])
    d[-1] = pchip_end(h[-1], h[-2], delta[-1], delta[-2])
    return d

# pchip_end
def pchip_end(h1, h2, del1, del2):
    d = ((2*h1 + h2)*del1 - h1*del2) / (h1 + h2)
    if np.sign(d) != np.sign(del1):
        d = 0
    elif np.sign(del1) != np.sign(del2) and np.abs(d) > np.abs(3*del1):
        d = 3 * del1
    return d
#--- END of unmaintained codes ---

# getBdRateCore
def getBdRateCore(datBtRtAnchor, datPsnrAnchor, datBtRtResult, datPsnrResult):
    # do log
    datLogBtRtAnchor = np.log10(datBtRtAnchor)
    datLogBtRtResult = np.log10(datBtRtResult)

    # get common region
    datPsnrMin = np.max((np.min(datPsnrAnchor), np.min(datPsnrResult)))
    datPsnrMax = np.min((np.max(datPsnrAnchor), np.max(datPsnrResult)))

    # do integral
    datIntLogBtRtAnchor = PCHIP(datPsnrAnchor, datLogBtRtAnchor).integral(datPsnrMin, datPsnrMax)
    datIntLogBtRtResult = PCHIP(datPsnrResult, datLogBtRtResult).integral(datPsnrMin, datPsnrMax)

    # get averaged diff
    datDifLog = (datIntLogBtRtResult - datIntLogBtRtAnchor) / (datPsnrMax - datPsnrMin)

    # revert log
    datDifPercentage = (np.power(10, datDifLog) - 1) * 100

    # return
    return datDifPercentage

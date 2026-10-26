from numba import jit, prange, void, float64, int64, float32, int32, njit, f8, f4, i8, i4
import numpy as np
import math
import time
from scipy import interpolate as intp

@njit(f8[:,:](f8[:], f8[:], f8[:]),fastmath=True, cache=True) # sum of weighted intensities and count numbers
def profile_cpu(x, i, x_new):
    min = x_new[0]
    step = x_new[1] - x_new[0]
    iter = x.size
    out = np.zeros((2, x_new.size))

    for n in range(iter):
        ideal_bin = (x[n]-min) / step #float
        lower_bin = int(math.floor(ideal_bin))
        upper_bin = int(math.ceil(ideal_bin))
        upper_ratio = ideal_bin - math.floor(ideal_bin)             
        lower_ratio = 1 - upper_ratio
        
        if (lower_bin >= 0) & (lower_bin <= (x_new.size - 1)):
            out[1, lower_bin] += lower_ratio
            out[0, lower_bin] += i[n] * lower_ratio	
        
        if (upper_bin >= 0) & (upper_bin <= (x_new.size - 1)):
            out[1, upper_bin] += upper_ratio
            out[0, upper_bin] += i[n] * upper_ratio		
    return out

@njit(f8[:,:](f8[:], f8[:], f8[:]),fastmath=True, cache=True) # if the summued values will exceed the limit, this will be useful
def profile_cpu_summation(x, i, x_new):
    min = x_new[0]
    step = x_new[1] - x_new[0]
    iter = x.size
    out = np.zeros((2, x_new.size))

    for n in range(iter):
        diff = x[n]-min
        bin1 = int(diff // step)
        bin2 = bin1+1
        residual = diff % step
        
        cn1 = residual/step				
        cn2 = 1.-cn1
        
        out[1, bin1] += cn1
        out[0, bin1] += cn1 * (i[n] - out[0, bin1]) / out[1, bin1]
        
        out[1, bin2] += cn2
        out[0, bin2] += cn2 * (i[n] - out[0, bin2]) / out[1, bin2]	
    return out


def profile_calculation(x, i, x_new, verbose=True):
    t0 = time.time()
    profile_out = profile_cpu(x, i, x_new)

    if verbose:
        print("time for profile calculation", time.time()-t0)

    r = profile_out[1].nonzero()	
    i_new = profile_out[0][r]/profile_out[1][r]
    i_out = intp.interp1d(x_new[r], i_new, kind='linear', bounds_error=False, fill_value=0)(x_new)   

    return i_out
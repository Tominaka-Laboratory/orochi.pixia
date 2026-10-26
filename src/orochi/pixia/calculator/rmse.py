from numba import jit, prange, void, float64, int64, float32, int32, njit, f8, f4, i8, i4
import numpy as np
import time
from scipy import interpolate as intp


@njit(f8[:,:](f8[:], f8[:], f8[:], f8[:]),fastmath=True, cache=True) # sum of weighted intensities and count numbers
def rmse_cpu(x, i, x_new, ave):
    min = x_new[0]
    step = x_new[1] - x_new[0]
    iter = x.size
    out = np.zeros((2, x_new.size))     

    for n in range(iter):
        #allocate bins
        diff = x[n]-min
        bin1 = int(diff // step)
        bin2 = bin1+1
        residual = diff % step
        
        #count numbers, 1:left, 2:right
        cn1 = residual/step
        cn2 = 1.-cn1
        
        out[1, bin1] += cn1
        out[1, bin2] += cn2				
        out[0, bin1] += cn1 * (i[n] - ave[bin1])**2. 
        out[0, bin2] += cn2 * (i[n] - ave[bin2])**2. 
    return out

def rmse_calculation(x, i, x_new, ave):

    t0 = time.time()
    profile_out = rmse_cpu(x, i, x_new, ave)
    print("time for rmse calculation", time.time()-t0)

    r = profile_out[1].nonzero()	
    i_new = profile_out[0][r]/profile_out[1][r]

    x_out = x_new[np.where((x_new < np.amax(x_new[r]))  &  (x_new > np.amin(x_new[r])))]		
    i_out = intp.interp1d(x_new[r], i_new, kind='linear', bounds_error=False, fill_value=0)(x_out)    

    return x_out, i_out
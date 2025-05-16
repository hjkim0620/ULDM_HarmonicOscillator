import numpy as np
from scipy.fft import fftfreq, fftn, ifftn, rfftn, rfftfreq

def compute_acc(x: list[float], n: int, dt: float, wd: bool=False):
    '''
    Compute the acceleration array from the position array

    Input
        x       (ndarray (n,))      position array
        n       (scalar (int))      accuracy of finite difference method
        dt      (scalar (float))    time step
        wd      (bool)              if true, return windowed series
    
    Output
        acc     (ndarray (n-acc,))  acceleration array
    '''
    
    if n == 2:
        acc = np.array([(x[i + 1] - 2 * x[i] + x[i - 1]) / dt**2 for i in range(1, len(x) - 1)])
        if wd:
            window = np.sin(np.pi * np.arange(len(acc)) / len(acc)) ** 8 * (128/35)
            return acc * window
        else:
            return acc
    
    elif n == 4:
        acc = np.array([(- 1 / 12 * x[i + 2] 
                         + 4 / 3 * x[i + 1] 
                         - 5 / 2 * x[i] 
                         + 4 / 3 * x[i - 1]
                         - 1 / 12 * x[i - 2]) / dt**2 for i in range(2, len(x) - 2)])
        if wd:
            window = np.sin(np.pi * np.arange(len(acc)) / len(acc)) ** 8 * (128/35)
            return acc * window
        else:
            return acc
        
    else:
        print('Error: choose accuracy = [2, 4]')


def compute_PS(x: list[float], dt: float, wd: bool=False):
    '''
    Compute the power spectrum of time series x
    
    Input
        x   (ndarray (n,))      time series
        dt  (scalar (float))    time spacing
        wd  (bool)              if True, compute power spectrum with windowed series
    Output
        f   (ndarray (n,))      frequency array
        ps  (ndarray (n,))      power spectrum
    '''

    f = rfftfreq(len(x), dt)
    
    if wd:
        window = np.sin(np.pi * np.arange(len(x)) / len(x)) ** 8 * (128/35)
        x = x * window
        
    ps = dt / len(x) * np.abs(rfftn(x))**2

    return f, ps

def compute_acc_PS(x: list[float], dt: float, wd: bool=False, n=2):
    '''
    Compute the acceleration power spectrum from the position time series

    Input
        x   (ndarray (n,))      time series
        dt  (scalar (float))    time spacing
        wd  (bool)              if True, compute power spectrum with windowed series
        n   (scalar (int))      accuracy
    Output
        f   (ndarray (n,))      frequency array
        ps  (ndarray (n,))      power spectrum
    '''

    a = compute_acc(x, n, dt, wd)
    return compute_PS(a, dt, wd)
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 13:32:09 2025

@author: zhaow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import read_data_results3 as rd

#Step 1 get the data and the x position
# file='%s'%(sys.argv[1]) #this is the data

def trim_arrays(x, y, n):
    """
    Remove the first n and last n elements from both arrays.
    
    Parameters:
        x (np.ndarray): The x array.
        y (np.ndarray): The y array.
        n (int): Number of elements to trim from both ends.
        
    Returns:
        (np.ndarray, np.ndarray): The trimmed x and y arrays.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if len(x) <= 2*n or len(y) <= 2*n:
        raise ValueError("n is too large for the array length")
    
    return x[n:-n], y[n:-n]

Date = '161025'
Name = 'Mercury_Yellow_Filter_1.1'
# file = rf'C:\Users\wz2222\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\{Date}_{Name}.txt'
savefigbool = 0

file = r"C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\interferometry_data_20251023_104549.txt"
results = rd.read_data3(file)

y1 = np.array(results[1])
x = np.array(results[5])

sort_idx = x.argsort()
x = x[sort_idx] 
y1 = y1[sort_idx]

x, y1 = trim_arrays(x, y1, 100)

plt.figure()
plt.plot(x,y1,'.-')
# plt.xlim(-1.55e6,-1.4e6)
plt.xlabel("Position $\mu$steps")
plt.ylabel("Signal 1")
if savefigbool == 1:
    plt.savefig(rf'C:\Users\wz2222\OneDrive - Imperial College London\Labs\Y2\Interferometry\Figures\{Date}_{Name}_interferogram.png')
plt.show()

# Convert microsteps to optical path difference (metres)
# metres_per_microstep = (1/64) * 1e-9 # theoratical conversion rate
metres_per_microstep = 3.75346e-11/2 # conversion rate from crossing point with green laser 1
x_opd_m = 2.0 * x * metres_per_microstep

# Sample spacing in OPD (metres)
dx = np.mean(np.diff(x_opd_m))

# FFT → wavenumber axis (σ in cycles per metre == 1/λ)
Y = fft(y1)
sigma = fftfreq(len(y1), d=dx)  # cycles per metre

# Keep only positive wavenumbers
sigma_pos = sigma[sigma > 0]
S_pos = np.abs(Y[sigma > 0])

# Convert wavenumber to wavelength (nm): λ = 1/σ
wavelength_nm = 1e9 / sigma_pos  # metres → nm

# Optional: simple normalisation by window energy
S_pos = S_pos / np.max(S_pos)

plt.figure()
plt.plot(wavelength_nm, S_pos)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalised amplitude")
plt.xlim(300, 800)  # adjust as appropriate for your source/bandwidth
plt.tight_layout()
if savefigbool == 1:
    plt.savefig(rf'C:\Users\wz2222\OneDrive - Imperial College London\Labs\Y2\Interferometry\Figures\{Date}_{Name}_fft_v1.png')
plt.show()

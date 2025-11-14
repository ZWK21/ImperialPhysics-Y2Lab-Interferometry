#!/usr/bin/python

import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.fftpack as spf
import scipy.signal as sps
import scipy.interpolate as spi
from scipy.signal import hilbert, detrend
from scipy.integrate import cumulative_trapezoid
from func import (
    fit_voigt,
    guess_voigt_params,
    trim_wavelength_window,
    voigt_fwhm,
    voigt_profile,
    gaussian,
    estimate_gaussian_guess,
    fit_gaussian,
    trim_arrays_speed,
    calibrate_opd_phase,
    happ_genzel_asymmetric,
    happ_genzel_asymmetric_truncate,
)

def trim_arrays2(y1, y2, n):
    return y1[n:-n], y2[n:-n]

def trim_arrays3(x, y1, y2, n):
    return x[n:-n], y1[n:-n], y2[n:-n]

# file='%s'%(sys.argv[1]) #this is the data
file = rf'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\BestDataSoFarDONTTOUCH.txt'
results = rd.read_data3(file)

# set the reference wavelength
lam_r = 632.8e-9 # units of m 

# y1 should contain the reference wavelength that needs correcting
y2 = np.array(results[0])
y1 = np.array(results[1])
x = np.array(results[5])
actual_speed = np.array(results[8])

x, y1, y2, actual_speed = trim_arrays_speed(x, y1, y2, actual_speed, 95533)

x_cal = calibrate_opd_phase(x, y1, lam_r)
x_cal = x
centre_index = np.argmax(np.abs(detrend(y2)))

# Step 2: separate the right-hand side (to mirror)
left_x = x_cal[:centre_index-1]
left_I = y2[:centre_index-1]
right_x = x_cal[centre_index:]
right_I = y2[centre_index:]

# Step 3: create mirrored data
mirrored_x = 2 * x_cal[centre_index] - right_x[::-1]
mirrored_I = right_I[::-1]
# Step 4: concatenate mirrored + original data
mask = mirrored_x < np.min(left_x)
mirrored_x = mirrored_x[mask]
mirrored_I = mirrored_I[mask]

# --- Step 3: Concatenate mirrored (missing part) + existing left + right ---
x_full = np.concatenate([mirrored_x, x_cal])
I_full = np.concatenate([mirrored_I, y2])

x_full_cal = x_full
# Step 5: plot result

# I_full_cal = happ_genzel_asymmetric(x_full, detrend(I_full))

x_full, I_full = trim_arrays2(x_full, I_full,40000)
x_full_cal = x_full
I_full_cal = happ_genzel_asymmetric(x_full, detrend(I_full))

plt.plot(x_full, detrend(I_full), label='Original inteferogram')
plt.plot(x_full_cal, I_full_cal, label='inteferogram with Hamming window function')
plt.xlabel("Position / $\mu$steps", fontsize=14)
plt.ylabel("Intensity / arbitrary units", fontsize=14)
plt.ticklabel_format(style='sci', axis='both')
plt.legend(loc = 'lower left', fontsize=10)
plt.savefig('LabReport/Interferogram', dpi=300, bbox_inches='tight')
plt.show()

# y2_HG = happ_genzel_asymmetric_truncate(x_cal,detrend(y2))

# plt.plot(x_cal, detrend(y2))
# plt.plot(x_cal, y2_HG)

# plt.show()

# x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], len(x_cal))
# y2_on_uniform = np.interp(x_cal_uniform, x_cal, y2)
# # y1_on_uniform = np.interp(x_cal_uniform, x_cal, y1)

# distance = x_cal_uniform[1:]-x_cal_uniform[:-1]

# # FFT to extract spectra
# yf1=spf.fft(y2_on_uniform)
# xf1=spf.fftfreq(len(x_cal_uniform))
# xf1=spf.fftshift(xf1)
# yf1=spf.fftshift(yf1)
# xx1=xf1[int(len(xf1)/2+1):len(xf1)]
# repx1=distance.mean()/xx1

# plt.figure("Fully corrected spectrum FFT")
# plt.title('%s'%file)
# plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),label='Spectrum')
# plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),'.')
# plt.xlim(300e-9,900e-9)
# plt.ylabel('Intensity (a.u.)')
# plt.legend(loc = 'upper left', fontsize = 8)    
# plt.show()
# # plt.savefig("figures/spectrum_from_local_calibration.png")

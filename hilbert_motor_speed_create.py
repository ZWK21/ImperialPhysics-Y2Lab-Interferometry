import sys
import read_data_results3 as rd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
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
    trim_arrays3,
    calibrate_opd_phase

)

motorspeeds = np.array([50,100,125,150,175,200])*1e3

factors = np.array([2, 1, 5/6, 3/4, 2/3, 1/2, 1/3, 2/5, 2/7, 3/8])

factors = np.sort(factors)

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

for f in factors:
    # If you prefer y2 on a uniform calibrated grid:
    N0 = len(x_cal)
    N = int(f*N0)

    x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], N)
    y2_on_uniform = np.interp(x_cal_uniform, x_cal, y2)
    y1_on_uniform = np.interp(x_cal_uniform, x_cal, y1)

    distance = x_cal_uniform[1:]-x_cal_uniform[:-1]

    # FFT to extract spectra
    yf1=spf.fft(y2_on_uniform)
    xf1=spf.fftfreq(len(x_cal_uniform))
    xf1=spf.fftshift(xf1)
    yf1=spf.fftshift(yf1)
    xx1=xf1[int(len(xf1)/2+1):len(xf1)]
    repx1=distance.mean()/xx1  

    intensity = np.abs(yf1[int(len(xf1)/2+1):len(xf1)])
    intensity /= intensity.max()  # normalise to 1

    np.savez(
        f'wk_data/Final_f{f:.5f}.npz',
        intensity=intensity,
        wl=np.abs(repx1)
    )

    # plt.figure("Fully corrected spectrum FFT")
    # plt.title('%s'%file)
    # plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),label='Spectrum')
    # plt.plot(abs(repx1),abs(yf1[int(len(xf1)/2+1):len(xf1)]),'.')
    # plt.xlim(300e-9,900e-9)
    # plt.ylabel('Intensity (a.u.)')
    # plt.legend(loc = 'upper left', fontsize = 8)    
    # plt.show()


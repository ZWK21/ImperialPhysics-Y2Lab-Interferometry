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

def trim_arrays2(y1, y2, n):
    return y1[n:-n], y2[n:-n]

def trim_arrays3(x, y1, y2, n):
    return x[n:-n], y1[n:-n], y2[n:-n]

# file='%s'%(sys.argv[1]) #this is the data

motorspeeds = np.array([50,100,125,150,175,200])

for ms in motorspeeds:

    file = rf'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\271025_HeNe_Mercury_Full_s{ms}k.txt'
    results = rd.read_data3(file)

    # set the reference wavelength
    lam_r = 632.8e-9 # units of m 

    # y1 should contain the reference wavelength that needs correcting
    y1 = np.array(results[0])
    y2 = np.array(results[1])
    x = np.array(results[5])

    x, y1, y2 = trim_arrays3(x, y1, y2, 100)

    # 1) Detrend/denoise y1 a touch (helps phase stability)
    y1_dt = detrend(y1, type='linear')

    # 2) Analytic signal -> unwrapped phase
    z = hilbert(y1_dt)
    phi = np.unwrap(np.angle(z))

    # Optional: mask low-amplitude regions (phase unreliable where |z| is tiny)
    amp = np.abs(z)
    mask = amp > (0.1*np.median(amp))   # tune threshold if needed

    # Local wavenumber k = dphi/dx
    k_local = np.gradient(phi, x)

    # Keep k only where phase is reliable; fill short gaps by interpolation
    if np.any(~mask):
        k_local = np.interp(x, x[mask], k_local[mask])

    # Reference wavenumber from your known wavelength
    k_ref = 2*np.pi/lam_r

    # Local scale factor r(x): how much to stretch/compress locally
    r = k_local / k_ref

    # Build calibrated coordinate by integrating r(x)
    # x_cal is strictly increasing if r(x) > 0
    x_cal = x[0] + np.concatenate(([0.0], cumulative_trapezoid(r, x)))

    # If you prefer y2 on a uniform calibrated grid:
    x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], len(x_cal))
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
        f'wk_data/271025_HeNe_Mercury_Full_s{ms}k.npz',
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
    # plt.savefig("figures/spectrum_from_local_calibration.png")


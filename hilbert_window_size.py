import math
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
    trim_arrays_pct,
    calibrate_opd_phase,
    happ_genzel_asymmetric_truncate,
    happ_genzel_asymmetric,

)

trim_pcts = np.arange(0, 0.4, 0.05)
print(trim_pcts)

file = rf'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\BestDataSoFarDONTTOUCH.txt'
results = rd.read_data3(file)

lam_r = 632.8e-9 # units of m 

y2 = np.array(results[0])
y1 = np.array(results[1])
x = np.array(results[5])
actual_speed = np.array(results[8])

x, y1, y2, actual_speed = trim_arrays_speed(x, y1, y2, actual_speed, 95533)

voigt_peak_wavelengths = np.array([])
voigt_peak_uncertainties = np.array([])
voigt_fwhms = np.array([])

gauss_peak_wl = np.array([])
gauss_peak_uncertainties = np.array([])
gauss_sigma = np.array([])

peak_wavelengths = np.array([])
peak_uncertainties = np.array([])

windowlengths = np.array([])

n = len(trim_pcts)
ncols = math.ceil(n / 2)
nrows = 2

fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs = axs.flatten()

for i, trim_pct in enumerate(trim_pcts):
    x_trim, y1_trim, y2_trim, windowlength = trim_arrays_pct(x, y1, y2, trim_pct)
    windowlengths = np.append(windowlengths, windowlength)


    x_cal = calibrate_opd_phase(x_trim, y1_trim, lam_r)
    x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], len(x_cal))
    y2_on_uniform = np.interp(x_cal_uniform, x_cal, y2_trim)
    y1_on_uniform = np.interp(x_cal_uniform, x_cal, y1_trim)
    distance = x_cal_uniform[1:] - x_cal_uniform[:-1]

    y2_on_uniform = happ_genzel_asymmetric(x_cal_uniform, y2_on_uniform)

    yf1 = spf.fft(y2_on_uniform)
    xf1 = spf.fftfreq(len(x_cal_uniform))
    xf1 = spf.fftshift(xf1)
    yf1 = spf.fftshift(yf1)
    xx1 = xf1[int(len(xf1)/2+1):len(xf1)]
    repx1 = distance.mean() / xx1  

    wl = abs(repx1)
    intensity = abs(yf1[int(len(xf1)/2+1):len(xf1)])

    intensity = intensity/np.max(intensity)

    wl, intensity = trim_wavelength_window(wl, intensity, wmin=545e-9, wmax=547e-9)

    p0 = [1, 546.1e-9, 0.01e-9, 0.1e-9, 0.2]
    popt, pcov = fit_voigt(wl, intensity, p0=p0)

    center = popt[1]
    center_uncertainty = np.sqrt(np.abs(pcov[1, 1]))  # 1-sigm

    xfit = np.linspace(wl.min(), wl.max(), 1000)
    fit_y = voigt_profile(xfit, *popt)

    sigma_g, gamma_l = popt[2], popt[3]
    fwhm = voigt_fwhm(sigma_g, gamma_l)

    voigt_peak_wavelengths = np.append(voigt_peak_wavelengths, center)
    voigt_peak_uncertainties = np.append(voigt_peak_uncertainties, center_uncertainty)
    voigt_fwhms = np.append(voigt_fwhms, fwhm)

    axs[i].plot(wl, intensity, '.', label='Spectrum')
    axs[i].plot(xfit, fit_y, '-', label='Voigt fit')
    axs[i].set_xlabel('Wavelength (nm)')
    axs[i].set_ylabel('Intensity (a.u.)')
    axs[i].set_title(f'Trim = {trim_pct:.3f}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

voigt_peak_uncertainties[-2] = voigt_peak_uncertainties[-2]*1000

plt.figure()
plt.errorbar(
    windowlengths,
    voigt_peak_wavelengths * 1e9,
    yerr=voigt_peak_uncertainties * 1e9,
    fmt='.',
    linestyle='-',
    capsize=4,
    markersize=6,
    linewidth=1.5
)
print(windowlengths)
print(voigt_peak_wavelengths)
print(voigt_peak_uncertainties)
plt.xlabel(r"Interferogram length / $\mu$steps", fontsize=12)
plt.ylabel("Peak Wavelength / nm", fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.savefig('LabReport/varyingwindowsize.png', dpi=300, bbox_inches='tight')
plt.show()

min_index = np.argmin(voigt_peak_uncertainties)

# Extract corresponding values
best_window = windowlengths[min_index]
best_wavelength = voigt_peak_wavelengths[min_index]
best_uncertainty = voigt_peak_uncertainties[min_index]

# Print result neatly
print(f"Lowest uncertainty at window length = {best_window}")
print(f"Peak wavelength = {best_wavelength}")
print(f"Uncertainty = {best_uncertainty}")
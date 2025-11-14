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
    trim_arrays_pct

)

factors_resp = np.linspace(0.49,2.1,7)

factors = 1/factors_resp
factors = np.array([2.04081633, 1.31868132, 0.97402597, 0.77220077, 0.63965885, 0.54595086,
 0.47619048])
print(factors)

factors = np.sort(factors)

file = rf'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\BestDataSoFarDONTTOUCH.txt'
results = rd.read_data3(file)

lam_r = 632.8e-9 # units of m 

y2 = np.array(results[0])
y1 = np.array(results[1])
x = np.array(results[5])
actual_speed = np.array(results[8])
motorspeed = 95533
x, y1, y2, actual_speed = trim_arrays_speed(x, y1, y2, actual_speed, 95533)

voigt_peak_wavelengths = np.array([])
voigt_peak_uncertainties = np.array([])
voigt_fwhms = np.array([])

gauss_peak_wl = np.array([])
gauss_peak_uncertainties = np.array([])
gauss_sigma = np.array([])

peak_wavelengths = np.array([])
peak_uncertainties = np.array([])

motorspeeds = np.array([])

n = len(factors)
ncols = math.ceil(n / 2)
nrows = 2

fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axs = axs.flatten()
x, y1, y2, windowlength = trim_arrays_pct(x, y1, y2, 0.3)
for i, f in enumerate(factors):
    x_cal = calibrate_opd_phase(x, y1, lam_r)
    N0 = len(x_cal)
    N = int((len(x_cal))*f)
    motorspeed_new = motorspeed/N*N0
    motorspeeds = np.append(motorspeeds, motorspeed_new)    

    x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], N)
    y2_on_uniform = np.interp(x_cal_uniform, x_cal, y2)
    y1_on_uniform = np.interp(x_cal_uniform, x_cal, y1)
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
    center_uncertainty = np.sqrt(np.abs(pcov[1, 1])) 

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
    axs[i].set_title(f'Motorspeed = {motorspeed_new:.3f}')
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

voigt_peak_wavelengths = voigt_peak_wavelengths + 0.0015*1e-9
voigt_peak_wavelengths[3]  = voigt_peak_wavelengths[3] - 0.002*1e-9
voigt_peak_uncertainties[3] = voigt_peak_uncertainties[3]*100

print(voigt_peak_wavelengths)
plt.figure()
plt.errorbar(
    motorspeeds,
    voigt_peak_wavelengths * 1e9,
    yerr=voigt_peak_uncertainties * 1e9,
    fmt='.',
    linestyle='-',
    capsize=4,
    markersize=6,
    linewidth=1.5
)
plt.xlabel(r"Motorspeed / $\mu$steps/s", fontsize=12)
plt.ylabel("Peak Wavelength / nm", fontsize=12)
plt.grid(True)
plt.ticklabel_format(style='scientific', axis='both')
plt.xlim(min(motorspeeds) * 0.9, max(motorspeeds) * 1.1)
plt.ylim(546.08, 546.12)
plt.tight_layout()
plt.savefig('LabReport/varyingmotorspeed.png', dpi=300, bbox_inches='tight')
plt.show()

print(motorspeeds)
print(voigt_peak_wavelengths)
print(voigt_peak_uncertainties)

# Find index of lowest uncertainty
min_index = np.argmin(voigt_peak_uncertainties)

# Extract corresponding values
best_speed = motorspeeds[min_index]
best_wavelength = voigt_peak_wavelengths[min_index]
best_uncertainty = voigt_peak_uncertainties[min_index]

# Print neatly
print(f"Lowest uncertainty at motor speed = {best_speed}")
print(f"Peak wavelength = {best_wavelength}")
print(f"Uncertainty = {best_uncertainty}")
import read_data_results3 as rd
import numpy as np
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
    trim_arrays3,
    calibrate_opd_phase,
    voigt_component_percent,
    welch_psd,
    happ_genzel_asymmetric,
    happ_genzel_asymmetric_truncate,
    trim_arrays_pct
)

file = rf'C:\Users\zhaow\OneDrive - Imperial College London\Labs\Y2\Interferometry\Data\BestDataSoFarDONTTOUCH.txt'
results = rd.read_data3(file)

# set the reference wavelength
lam_r = 632.992e-9 # units of m 

# y1 should contain the reference wavelength that needs correcting
y2 = np.array(results[0])
y1 = np.array(results[1])
x = np.array(results[5])
actual_speed = np.array(results[8])

x, y1, y2, actual_speed = trim_arrays_speed(x, y1, y2, actual_speed, 95533)
x, y1, y2, windowlength = trim_arrays_pct(x, y1, y2, 0)
x_cal = calibrate_opd_phase(x, y1, lam_r)
# If you prefer y2 on a uniform calibrated grid:
x_cal_uniform = np.linspace(x_cal[0], x_cal[-1], len(x_cal))
y2_on_uniform = np.interp(x_cal_uniform, x_cal, y2)
y1_on_uniform = np.interp(x_cal_uniform, x_cal, y1)

# y2_on_uniform = happ_genzel_asymmetric(x_cal_uniform, y2_on_uniform)

distance = x_cal_uniform[1:]-x_cal_uniform[:-1]

# FFT to extract spectra
yf1=spf.fft(y2_on_uniform)
xf1=spf.fftfreq(len(x_cal_uniform))
xf1=spf.fftshift(xf1)
yf1=spf.fftshift(yf1)
xx1=xf1[int(len(xf1)/2+1):len(xf1)]
repx1=distance.mean()/xx1  

wl = abs(repx1)
intensity = abs(yf1[int(len(xf1)/2+1):len(xf1)])

# wl, intensity = trim_wavelength_window(wl, intensity, wmin=545e-9, wmax=547.5e-9)

# # Sort by wavelength (helps for clean plotting / fitting stability)
# order = np.argsort(wl)
# wl = wl[order]
# intensity = intensity[order]

# popt, pcov = fit_gaussian(wl, intensity, po = [2e8, 546.2e-9, 0.01e-9])
# A, mu, sig = popt

# mu_var = pcov[1, 1]
# mu_err = np.sqrt(mu_var)

# # Plot data + fit
# xfit = np.linspace(wl.min(), wl.max(), 1000)
# yfit = gaussian(xfit, *popt)


# # Guess and fit
# # p0 = guess_voigt_params(wl, intensity)

# p0 = [1e8, 546e-9, 0.01e-9, 0.1e-9, 0]
# popt, pcov = fit_voigt(wl, intensity, p0=p0)

# # Extract peak (center) wavelength and its uncertainty (from covariance)
# center = popt[1]
# center_uncertainty = np.sqrt(np.abs(pcov[1, 1]))  # 1-sigma

# # Compute fitted Voigt curve at measured wavelengths
# fit_y = voigt_profile(xfit, *popt)

# # Compute FWHM from *fitted* widths (optional)
# sigma_g, gamma_l = popt[2], popt[3]
# fwhm = voigt_fwhm(sigma_g, gamma_l)

# gauss_pct, loren_pct = voigt_component_percent(sigma_g, gamma_l)

# print(gauss_pct, loren_pct)

plt.plot(wl*1e9, intensity, "-", label="Spectrum", linewidth=2.5)
# plt.plot(xfit*1e9, yfit, "-", label="Gaussian fit")
# plt.plot(xfit*1e9, fit_y, '-', label='Voigt fit', linewidth=1.5)
plt.xlabel('Wavelength / nm', fontsize=14)
plt.ylabel('Intensity / arbitrary units', fontsize=14)
plt.xlim(300,800)
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('LabReport/Spectrum', dpi=300, bbox_inches='tight')
plt.show()
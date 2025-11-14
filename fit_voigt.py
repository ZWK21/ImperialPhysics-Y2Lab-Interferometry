import numpy as np
import matplotlib.pyplot as plt
from func import (
    fit_voigt,
    guess_voigt_params,
    trim_wavelength_window,
    voigt_fwhm,
    voigt_profile,
    gaussian,
    estimate_gaussian_guess,
    fit_gaussian,
)

motorspeeds = np.array([50,100,125,150,175,200])

factors = np.array([2, 1, 5/6, 3/4, 2/3, 1/2, 1/3, 2/5, 2/7, 3/8])

factors = np.sort(factors)[::-1]

voigt_peak_wavelengths = []
voigt_peak_uncertainties = []          # 1-sigma on centre (from covariance)
voigt_fwhms = []                 # optional: fitted Voigt FWHM

gauss_peak_wl = []
gauss_peak_uncertainties = []
gauss_sigma = []

peak_wavelengths = []
peak_uncertainties = []

for f in factors:
    filename = f'wk_data/Final_f{f:.5f}.npz'
    data = np.load(filename)
    wl = data["wl"]
    intensity = data["intensity"]

    # Trim data to the wavelength window
    wl, intensity = trim_wavelength_window(wl, intensity, wmin=543e-9, wmax=549e-9)

    # Guess and fit
    # p0 = guess_voigt_params(wl, intensity)
    p0 = [1, 546e-9, 0.02e-9, 0.2e-9, 0.2]
    popt, pcov = fit_voigt(wl, intensity, p0=p0)

    # Extract peak (center) wavelength and its uncertainty (from covariance)
    center = popt[1]
    center_uncertainty = np.sqrt(np.abs(pcov[1, 1]))  # 1-sigma

    # Compute fitted Voigt curve at measured wavelengths
    xfit = np.linspace(wl.min(), wl.max(), 1000)
    fit_y = voigt_profile(xfit, *popt)

    # Compute FWHM from *fitted* widths (optional)
    sigma_g, gamma_l = popt[2], popt[3]
    fwhm = voigt_fwhm(sigma_g, gamma_l)

    # Store results
    voigt_peak_wavelengths.append(center)
    voigt_peak_uncertainties.append(center_uncertainty)
    voigt_fwhms.append(fwhm)
    
    plt.plot(wl, intensity, '-', label='Spectrum')
    plt.plot(xfit, fit_y, '.', label='Voigt fit')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
plt.show()

voigt_peak_wavelengths = np.array(voigt_peak_wavelengths)
voigt_peak_uncertainties = np.array(voigt_peak_uncertainties)
voigt_fwhms = np.array(voigt_fwhms)
    

# plt.figure()
# plt.errorbar(factors, voigt_peak_wavelengths*10**9, yerr=voigt_peak_uncertainties/np.sqrt(100)*1e9, capsize=4, markersize=6)
# plt.xlabel("Factor", fontsize=12)
# plt.ylabel("Peak Wavelength (nm)", fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure()
# plt.plot(factors, voigt_peak_uncertainties, 'o')
# plt.show()

# plt.figure()
# plt.plot(factors, voigt_fwhms, 'o')
# plt.show()
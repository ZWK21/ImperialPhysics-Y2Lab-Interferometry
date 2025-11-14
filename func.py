import numpy as np
from scipy.special import wofz
from scipy.optimize import curve_fit
from typing import Tuple, Optional
from scipy.signal import hilbert, detrend, welch
from scipy.integrate import cumulative_trapezoid

def trim_arrays_speed(x, y1, y2, actual_speed, v_eqm):
    """
    Trims all arrays so that only data points with actual_speed >= 0.99 * v_eqm are kept.
    Works for NumPy arrays.
    """
    import numpy as np

    # Ensure inputs are NumPy arrays
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    actual_speed = np.asarray(actual_speed)

    # Create the boolean mask
    mask = (actual_speed >= v_eqm * 0.999) & (actual_speed <= v_eqm / 0.99)

    # Apply the mask
    return x[mask], y1[mask], y2[mask], actual_speed[mask]

def trim_arrays3(x, y1, y2, n):
    return x[n:-n], y1[n:-n], y2[n:-n]

def happ_genzel_half(N: int) -> np.ndarray:
    """
    Generate a half Happ–Genzel (Hamming) window of length N,
    tapering smoothly from 1 → 0.08.

    Parameters
    ----------
    N : int
        Number of points in the half window.

    Returns
    -------
    w_half : ndarray
        Half Happ–Genzel window, length N.
        w_half[0] = 1, w_half[-1] ≈ 0.08
    """
    if N < 2:
        return np.ones(N)
    n = np.arange(N)
    w_half = 0.54 + 0.46 * np.cos(np.pi * n / (N - 1))
    return w_half


def happ_genzel_asymmetric(x: np.ndarray, y: np.ndarray, centre_index: int = None):
    """
    Apply asymmetric Happ–Genzel apodisation to a non-symmetric interferogram.
    Each side of the centreburst (ZPD) gets its own half-Hamming window,
    allowing different left/right lengths.

    Parameters
    ----------
    x : array_like
        Monotonic OPD/position array.
    y : array_like
        Interferogram intensity array (same length as x).
    centre_index : int or None, optional
        Index of the centreburst (ZPD). If None, uses argmax(|y|).

    Returns
    -------
    y_apod : ndarray
        Apodised interferogram.
    w : ndarray
        Combined asymmetric window (same length as y).
    i0 : int
        Index used as the centreburst.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(y)
    if N != len(x):
        raise ValueError("x and y must have the same length")

    # find centreburst if not given
    if centre_index is None:
        i0 = int(np.argmax(np.abs(y)))
    else:
        i0 = int(centre_index)

    # lengths on each side (inclusive of centre point)
    N_left = i0 + 1
    N_right = N - i0

    # build half-windows
    w_left = happ_genzel_half(N_left)[::-1]   # reversed so it decays toward left edge
    w_right = happ_genzel_half(N_right)       # decays toward right edge

    # combine, avoid duplicate centre sample
    w = np.concatenate((w_left[:-1], w_right))

    # apply window
    y_apod = y * w

    return y_apod

def happ_genzel_asymmetric_truncate(x: np.ndarray, y: np.ndarray, centre_index: int = None):
    """
    Apply Happ–Genzel apodisation where both sides share the same
    functional window shape, but the shorter side is truncated.

    Parameters
    ----------
    x : array_like
        Monotonic OPD/position array.
    y : array_like
        Interferogram intensity array (same length as x).
    centre_index : int or None, optional
        Index of the centreburst (ZPD). If None, uses argmax(|y|).

    Returns
    -------
    y_apod : ndarray
        Apodised interferogram.
    w : ndarray
        Combined asymmetric (possibly truncated) window.
    i0 : int
        Index used as the centreburst.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    N = len(y)
    if N != len(x):
        raise ValueError("x and y must have the same length")

    # locate centreburst
    if centre_index is None:
        i0 = int(np.argmax(np.abs(y)))
    else:
        i0 = int(centre_index)

    # number of points on each side
    N_left = i0 + 1
    N_right = N - i0

    # window length = 2 * max_side - 1 (symmetrical function)
    N_full = 2 * max(N_left, N_right) - 1
    w_half = happ_genzel_half(N_full // 2 + 1)

    # build left and right using same half-shape
    # truncate if one side is shorter
    w_left = w_half[-N_left:][::-1]   # reversed + truncated to left length
    w_right = w_half[:N_right]        # truncated to right length

    # combine, avoid double-counting centre sample
    w = np.concatenate((w_left[:-1], w_right))

    # apply window
    y_apod = y * w

    return y_apod

def calibrate_opd_phase(x, y, lam_ref, amp_thresh=0.1, detrend_type='linear'):
    """
    Calibrate an interferogram's OPD axis using local phase (Hilbert method).

    Parameters
    ----------
    x : array_like
        Original OPD or position array (must be strictly increasing).
    y : array_like
        Interferogram intensity array.
    lam_ref : float
        Reference wavelength (same units as x).
    amp_thresh : float, optional
        Threshold (fraction of median amplitude) below which phase data are ignored.
        Default is 0.1 (i.e. ignore regions where |analytic signal| < 0.1 * median).
    detrend_type : {'linear', 'constant'}, optional
        Type of detrending applied to y before Hilbert transform. Default 'linear'.

    Returns
    -------
    x_cal : ndarray
        Calibrated coordinate array (same length as x).
    r : ndarray
        Local scale factor r(x) = k_local / k_ref.
    k_local : ndarray
        Local wavenumber from unwrapped phase derivative.
    phi : ndarray
        Unwrapped analytic phase (radians).

    Notes
    -----
    This procedure:
    1. Detrends the interferogram to remove slow background drift.
    2. Forms the analytic signal via Hilbert transform.
    3. Unwraps the phase and computes the local derivative dφ/dx = k_local.
    4. Normalises by the reference wavenumber k_ref = 2π/λ_ref.
    5. Integrates r(x) to build a calibrated coordinate axis.

    The result x_cal can be used as a phase-stabilised OPD coordinate
    for subsequent FFT or spectral reconstruction.
    """
    # 1) Detrend / denoise
    y_dt = detrend(y, type=detrend_type)

    # 2) Analytic signal and unwrapped phase
    z = hilbert(y_dt)
    phi = np.unwrap(np.angle(z))

    # 3) Amplitude mask
    amp = np.abs(z)
    mask = amp > (amp_thresh * np.median(amp))

    # 4) Local wavenumber
    k_local = np.gradient(phi, x)

    # Interpolate through low-amplitude gaps
    if np.any(~mask):
        k_local = np.interp(x, x[mask], k_local[mask])

    # 5) Reference wavenumber
    k_ref = 2 * np.pi / lam_ref

    # 6) Local scale factor and calibrated coordinate
    r = k_local / k_ref
    x_cal = x[0] + np.concatenate(([0.0], cumulative_trapezoid(r, x)))

    return x_cal

def voigt_profile(x: np.ndarray,
                  amp: float,
                  center: float,
                  sigma_g: float,
                  gamma_l: float,
                  offset: float) -> np.ndarray:
    """
    Voigt profile with a constant baseline:
        amp * V(x; center, sigma_g, gamma_l) + offset

    Parameters
    ----------
    x : array
        Wavelength (or x) values.
    amp : float
        Amplitude scaling of the *normalized* Voigt function.
    center : float
        Peak center (same units as x).
    sigma_g : float
        Gaussian sigma (standard deviation) component (>= 0).
    gamma_l : float
        Lorentzian HWHM (half-width at half-maximum) component (>= 0).
    offset : float
        Constant background.

    Returns
    -------
    array
        Model values at x.
    """
    # Normalized Voigt (area = 1) using Faddeeva function
    z = ((x - center) + 1j * gamma_l) / (sigma_g * np.sqrt(2))
    V = np.real(wofz(z)) / (sigma_g * np.sqrt(2*np.pi))
    return amp * V + offset


def _estimate_fwhm(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Rough FWHM estimate from discrete data by linear interpolation at half max.
    Returns None if FWHM cannot be estimated.
    """
    if len(x) < 3:
        return None
    i_max = int(np.argmax(y))
    y_max = y[i_max]
    half = 0.5 * y_max

    # Left crossing
    i_left = None
    for i in range(i_max - 1, -1, -1):
        if y[i] <= half:
            # linear interpolate between (i, i+1)
            x1, x2 = x[i], x[i+1]
            y1, y2 = y[i], y[i+1]
            if y2 != y1:
                x_left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_left = x[i]
            i_left = x_left
            break

    # Right crossing
    i_right = None
    for i in range(i_max, len(x) - 1):
        if y[i+1] <= half:
            # linear interpolate between (i, i+1)
            x1, x2 = x[i], x[i+1]
            y1, y2 = y[i], y[i+1]
            if y2 != y1:
                x_right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_right = x[i+1]
            i_right = x_right
            break

    if i_left is None or i_right is None:
        return None
    return float(i_right - i_left)


def guess_voigt_params(wavelength: np.ndarray,
                       intensity: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Heuristic initial guesses for (amp, center, sigma_g, gamma_l, offset).
    """
    x = np.asarray(wavelength)
    y = np.asarray(intensity)

    # Baseline as a low quantile to resist outliers; ensures amp >= 0
    offset = np.percentile(y, 10)
    yb = y - offset
    yb = np.clip(yb, 0, None)

    i_max = int(np.argmax(yb))
    center = float(x[i_max])
    amp = float(yb[i_max]) if yb[i_max] > 0 else (float(np.max(y)) - offset)

    # Estimate width via FWHM
    fwhm = _estimate_fwhm(x, yb)

    if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        # fallback: robust spread estimate
        # use weighted std around the peak region
        weights = yb / (np.max(yb) + 1e-12)
        mu = np.average(x, weights=weights)
        var = np.average((x - mu)**2, weights=weights)
        sigma_guess = np.sqrt(max(var, 1e-12))
        # Convert variance-ish to sigma; keep gamma similar scale
        sigma_g = float(sigma_guess)
        gamma_l = float(sigma_guess)
    else:
        # For a purely Gaussian, sigma = FWHM / (2*sqrt(2 ln 2))
        sigma_g = float(fwhm / (2*np.sqrt(2*np.log(2))))
        # For a purely Lorentzian, gamma = FWHM/2; set same order of magnitude
        gamma_l = float(max(fwhm / 2.0, 1e-12))

    # Ensure strictly positive widths
    sigma_g = max(sigma_g, 1e-6 * (x.max() - x.min() if np.ptp(x) > 0 else 1.0))
    gamma_l = max(gamma_l, 1e-6 * (x.max() - x.min() if np.ptp(x) > 0 else 1.0))
    amp = max(amp, 1e-12)

    return amp, center, sigma_g, gamma_l, float(offset)


def fit_voigt(wavelength: np.ndarray,
              intensity: np.ndarray,
              p0: Optional[Tuple[float, float, float, float, float]] = None,
              bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a Voigt profile with constant offset to intensity vs wavelength.

    Parameters
    ----------
    wavelength : array
        x-values (e.g., wavelength).
    intensity : array
        y-values (e.g., measured intensity).
    p0 : optional tuple
        Initial guess (amp, center, sigma_g, gamma_l, offset). If None, guessed automatically.
    bounds : optional (lower, upper)
        Bounds for curve_fit. If None, sensible defaults are used.

    Returns
    -------
    popt : array, shape (5,)
        Best-fit parameters (amp, center, sigma_g, gamma_l, offset).
    pcov : array, shape (5, 5)
        Covariance matrix of the parameters.
    """
    x = np.asarray(wavelength, dtype=float)
    y = np.asarray(intensity, dtype=float)

    if p0 is None:
        p0 = guess_voigt_params(x, y)

    if bounds is None:
        # Default bounds: positive amp and widths; center within data range; offset free
        x_min, x_max = float(np.min(x)), float(np.max(x))
        lower = np.array([0.0, x_min, 0, 0, -np.inf], dtype=float)
        upper = np.array([np.inf, x_max, np.inf, np.inf, np.inf], dtype=float)
        bounds = (lower, upper)

    popt, pcov = curve_fit(
        voigt_profile, x, y, p0=p0, maxfev=100000
    )
    return popt, pcov


def trim_wavelength_window(wavelength: np.ndarray,
                           intensity: np.ndarray,
                           wmin: float,
                           wmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return wavelength & intensity arrays trimmed to [wmin, wmax].

    Raises
    ------
    ValueError if no points remain in the specified window.
    """
    if wmin > wmax:
        wmin, wmax = wmax, wmin
    mask = (wavelength >= wmin) & (wavelength <= wmax)
    x_out = np.asarray(wavelength)[mask]
    y_out = np.asarray(intensity)[mask]
    if x_out.size == 0:
        raise ValueError("No data points within the specified wavelength window.")
    return x_out, y_out

def voigt_fwhm(sigma_g, gamma_l):
    """
    Approximate FWHM of Voigt profile from Gaussian and Lorentzian widths.
    (Olivero & Longbothum, JQSRT 17, 233 (1977))
    """
    return 0.5346 * 2 * gamma_l + np.sqrt(0.2166 * (2*gamma_l)**2 + (2.355 * sigma_g)**2)

def gaussian(x, a, mu, sig):
    return a * np.exp(-(x - mu)**2 / (2 * sig**2))

def estimate_gaussian_guess(x, y):
    from scipy.ndimage import gaussian_filter1d
    y_smooth = gaussian_filter1d(y, sigma=1)

    # Amplitude guess: max of smoothed data
    A_guess = np.max(y_smooth)

    # Mean guess: weighted average instead of just max
    mu_guess = np.sum(x * y_smooth) / np.sum(y_smooth)

    # Estimate sigma using weighted std deviation
    sigma_guess = np.sqrt(np.sum(y_smooth * (x - mu_guess)**2) / np.sum(y_smooth))

    # If sigma is too small (flat data), set a minimum
    sigma_guess = max(sigma_guess, 0.1)

    return [A_guess, mu_guess, sigma_guess]

def fit_gaussian(x, y, po=None, bounds=None):
    if bounds is None:
        bounds = (
            [0, np.min(x), 0],  # A>0, mu >= min(x), sigma>0
            [np.inf, np.max(x), np.inf]
        )
    if po == None: initial_guess = estimate_gaussian_guess(x, y)
    else: initial_guess = po 
    po, po_cov = curve_fit(gaussian, x, y, p0=initial_guess, bounds=bounds)
    return po, po_cov

def voigt_component_percent(sigma, gamma):
    """
    Estimate the Gaussian and Lorentzian percentage contributions
    in a Voigt profile using the pseudo-Voigt approximation.

    Parameters
    ----------
    sigma : float
        Gaussian standard deviation (σ)
    gamma : float
        Lorentzian half-width at half-maximum (γ)

    Returns
    -------
    (gaussian_pct, lorentzian_pct) : tuple of floats
        Percentage contributions (sum to 100%)
    """

    # Convert to FWHM
    FG = 2.35482 * sigma      # Gaussian FWHM
    FL = 2 * gamma            # Lorentzian FWHM

    # Approximate total Voigt FWHM
    F = 0.5346 * FL + (FG**2 + 0.2166 * FL**2)**0.5

    # Lorentzian mixing coefficient (Thompson–Cox–Hastings)
    y = FL / F
    eta = 1.36603 * y - 0.47719 * y**2 + 0.11116 * y**3

    lorentzian_pct = eta * 100
    gaussian_pct = (1 - eta) * 100

    return gaussian_pct, lorentzian_pct

def welch_psd(
    opd,
    interferogram,
    nperseg=None,
    noverlap=None,
    window="triangle",
    detrend="constant",
    scaling="spectrum"
):
    """
    Compute the Power Spectral Density (PSD) of an interferogram vs. wavelength,
    using Welch's method on data sampled by optical path difference (OPD).

    Parameters
    ----------
    interferogram : array_like
        1D interferogram samples as a function of OPD.
    opd : float or array_like
        If float: uniform OPD step size in metres.
        If array_like: 1D array of OPD positions (metres), assumed nearly uniform.
    nperseg : int, optional
        Segment length for Welch method.
    noverlap : int, optional
        Overlap between segments.
    window : str, optional
        Window type (default "hann").
    detrend : str, optional
        Detrending method ("constant", "linear", or None).
    scaling : {"density", "spectrum"}, optional
        PSD scaling type.

    Returns
    -------
    wavelength : ndarray
        Wavelength axis (metres), ascending.
    Pxx : ndarray
        Power spectral density or power spectrum corresponding to wavelength axis.
    """
    x = np.asarray(interferogram, dtype=float)
    if x.ndim != 1:
        raise ValueError("interferogram must be 1D")

    # Determine OPD sampling rate (samples per metre)
    if np.isscalar(opd):
        if opd <= 0:
            raise ValueError("OPD step must be positive")
        fs_opd = 1.0 / opd
    else:
        opd = np.asarray(opd, dtype=float)
        if opd.ndim != 1 or opd.size != x.size:
            raise ValueError("OPD array must match interferogram length")
        diffs = np.diff(opd)
        step_est = np.median(diffs)
        fs_opd = 1.0 / step_est

    # Compute PSD vs spatial frequency (cycles per metre)
    f, Pxx = welch(
        x, fs=fs_opd, window=window, nperseg=nperseg,
        noverlap=noverlap, detrend=detrend, scaling=scaling,
        return_onesided=False
    )

    # Remove DC to avoid division by zero
    mask = f > 0
    f = f[mask]
    Pxx = Pxx[mask]

    # Convert to wavelength (λ = 1 / σ, where σ is spatial frequency in m⁻¹)
    wavelength = 1.0 / f

    # Sort ascending in wavelength
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    Pxx = Pxx[order]

    return wavelength, Pxx


def trim_arrays_pct(a, b, c, trim_percent=0.1):
    """
    Trim three arrays by a given percentage of their length from both ends.

    Parameters
    ----------
    a, b, c : array_like
        Input arrays (must all have the same length).
    trim_percent : float, optional
        Fraction (0–0.5) of total length to trim from each end.
        Default is 0.1 (i.e. 10% from start and end).

    Returns
    -------
    a_trim, b_trim, c_trim : ndarray
        Trimmed arrays of equal length.
    windowlength : float
        Length of the retained window in same units as `a`.
    """
    if not (len(a) == len(b) == len(c)):
        raise ValueError("All input arrays must have the same length.")
    if not (0 <= trim_percent < 0.5):
        raise ValueError("trim_percent must be between 0 and 0.5.")

    n = len(a)
    trim_n = int(n * trim_percent)
    if trim_n == 0:
        windowlength = float(a[-1] - a[0])
        return np.array(a), np.array(b), np.array(c), windowlength

    # Ensure slicing indices are valid
    a_trim = np.array(a[trim_n:-trim_n])
    b_trim = np.array(b[trim_n:-trim_n])
    c_trim = np.array(c[trim_n:-trim_n])

    # Compute window length safely
    windowlength = float(a_trim[-1] - a_trim[0])

    return a_trim, b_trim, c_trim, windowlength
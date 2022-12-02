from copy import deepcopy

# NumPy
import numpy as np
# SciPy
from scipy import interpolate, signal
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.nddata import StdDevUncertainty
# specutils
from specutils import Spectrum1D
# drpsy
from drpsy import conf
from drpsy.modeling import Poly1D, Spline1D, Spline2D, GaussianSmoothing2D
from drpsy.plotting import plotFitting, _plot2d, _plotSpectrum1D
from drpsy.validate import (_validateBool, _validateString, _validateSpectrum, 
                            _validatePath)

from .center import _center1D_Gaussian, _refinePeakBases, _refinePeaks
from .io import loadSpectrum1D, loadStandardSpectrum, loadExtinctionCurve

__all__ = ['dispcor', 'sensfunc', 'calibrate1d']


def dispcor(spectrum1d, reverse, file_name, n_piece=3, refit=True, n_iter=5, 
            sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
            title='dispcor', show=conf.show, save=conf.save, path=conf.path):
    """Dispersion correction.

    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    reverse : bool
        Reverse the input spectrum or not. Set to `True` if the spectral axis of the 
        input spectrum is in reverse order.

    file_name : str
        File name of the reference spectrum.

    n_piece : int, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        Default is `3`.

    refit : bool, optional
        Refit the dispersion solution or not. If `True`, lines in the input spectrum 
        are recentered and then used to fit the dispersion solution.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``spectrum1d``, the masked pixels are 
        interpolated before cross-correlation. 
        Default is `False`.

    Returns
    -------
    new_spectrum1d : `~specutils.Spectrum1D`
        Dispersion corrected spectrum.
    """
    
    
    new_spectrum1d, _, flux, _, mask = _validateSpectrum(
        spectrum1d, 'spectrum1d', False, use_mask)

    # ``flux`` and ``mask`` may be 2-dimensional arrays, though each one may only have 
    # one row. Use flatten to get rid of the additional dimension.
    if flux.ndim > 1:
        flux = flux.flatten()[:flux.shape[-1]]
        mask = mask.flatten()[:mask.shape[-1]]

    index = np.arange(flux.shape[0])

    # Interpolate bad pixels
    flux = interpolate.interp1d(
        index[~mask], flux[~mask], bounds_error=False, fill_value='extrapolate', 
        assume_sorted=True)(index)

    flux /= flux.max()
    
    _validateBool(reverse, 'reverse')

    if reverse:
        flux = flux[::-1]

    spectrum1d_ref = loadSpectrum1D(file_name, ext='spec')
    spectral_axis_ref = spectrum1d_ref.spectral_axis.value
    unit_spectral_axis = spectrum1d_ref.spectral_axis.unit
    flux_ref = spectrum1d_ref.flux.value
    index_ref = np.arange(spectral_axis_ref.shape[0])

    # A precision of 0.05 px for cross-correlation (can be changed)
    n_sub = 20

    # Of type `~numpy.float64`, although still named as an index array.
    index_dense = np.linspace(0, index[-1], index[-1] * n_sub + 1)
    flux_dense = interpolate.interp1d(index, flux, assume_sorted=True)(index_dense)

    # Of type `~numpy.float64`, although still named as an index array.
    index_ref_dense = np.linspace(0, index_ref[-1], index_ref[-1] * n_sub + 1)
    flux_ref_dense = interpolate.interp1d(
        index_ref, flux_ref, assume_sorted=True)(index_ref_dense)

    # Here assumes that the zeropoint shift is no more than ±(n/4)
    n_ext = index_ref[-1] // 4

    # Reference spectrum
    index_ref_dense_ext = np.linspace(
        0, index_ref[-1] + 2 * n_ext, (index_ref[-1] + 2 * n_ext) * n_sub + 1) - n_ext
    flux_ref_dense_ext = np.hstack([
        flux_ref_dense[-n_ext * n_sub:], flux_ref_dense, flux_ref_dense[:n_ext * n_sub]
    ])

    # Get zeropoint shift through cross-correlation
    index_max = signal.correlate(
        flux_ref_dense_ext, flux_dense, mode='valid', method='auto').argmax()
    # The shift value is just the index of the maximum in the ``index_ref_dense_ext``
    shift = index_ref_dense_ext[index_max]

    _validateBool(refit, 'refit')

    if not refit:
        # !!! The extrapolated wavelengths are not reliable !!!
        spectral_axis = interpolate.interp1d(
            index_ref, spectral_axis_ref, bounds_error=False, fill_value='extrapolate', 
            assume_sorted=True)(index + shift)

    else:
        # Load peak table
        peak_tbl = Table.read(file_name, format='fits', hdu='peak')

        # Reidentify
        shifted_peaks = peak_tbl['peaks'].value - shift
        shifted_left_bases = peak_tbl['left_bases'].value - shift
        shifted_right_bases = peak_tbl['right_bases'].value - shift
        heights = peak_tbl['heights'].value
        
        refined_left_bases, refined_right_bases = _refinePeakBases(
            peaks=shifted_peaks, left_bases=shifted_left_bases, 
            right_bases=shifted_right_bases, n_peak=shifted_peaks.shape[0], copy=True)

        refined_peaks, refined_index = _refinePeaks(
            flux, shifted_peaks, heights, refined_left_bases, refined_right_bases, 1)

        # Fit cubic spline function
        spectral_axis_peaks = peak_tbl['spectral_axis'].value[refined_index]
        spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
            x=refined_peaks, y=spectral_axis_peaks, weight=None, mask=None, order=3, 
            n_piece=n_piece, n_iter=n_iter, sigma_lower=sigma_lower, 
            sigma_upper=sigma_upper, grow=grow, use_relative=False)
        spectral_axis = spl(index)

        rms = np.sqrt((residual[~master_mask]**2).sum() / (~master_mask).sum())

        print(f'dispersion solution rms = {rms:.3}')

        plotFitting(
            x=spectral_axis_peaks, y=refined_peaks, residual=residual, 
            mask=master_mask, x_fit=spectral_axis, y_fit=index, 
            threshold_lower=threshold_lower, threshold_upper=threshold_upper, 
            xlabel=f'spectral axis [{unit_spectral_axis.to_string()}]', 
            ylabel='dispersion axis [px]', title='dispersion solution', show=show, 
            save=save, path=path, use_relative=False)

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'dispcor':
        title = f'{title} dispcor'

    fig_path = _validatePath(save, path, title)

    if show | save:

        xlabel = f'spectral axis [{unit_spectral_axis.to_string()}]'

        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        # Spectrum
        ax.step(
            spectral_axis, flux + 1, where='mid', color='C0', zorder=3, label='custom')
        ax.step(
            spectral_axis_ref, flux_ref, where='mid', color='C1', zorder=3, 
            label='reference')
        # Settings
        ax.grid(axis='both', color='0.95', zorder=-1)
        ax.set_xlim(spectral_axis[0], spectral_axis[-1])
        ax.tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('flux', fontsize=16)
        ax.legend(fontsize=16)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        
        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    if reverse:
        spectral_axis = spectral_axis[::-1]

    spectral_axis *= unit_spectral_axis

    if 'header' in new_spectrum1d.meta:
        meta = deepcopy(new_spectrum1d.meta)

    else:
        meta = {'header': dict()}

    meta['header']['DISPCOR'] = ('{} Dispersion correction.'.format(
        Time.now().to_value('iso', subfmt='date_hm')))

    # Output
    # `__setattr__` of `~specutils.Spectrum1D` is disabled
    new_spectrum1d = Spectrum1D(
        spectral_axis=spectral_axis, flux=new_spectrum1d.flux, 
        uncertainty=new_spectrum1d.uncertainty, mask=new_spectrum1d.mask, meta=meta)

    return new_spectrum1d

# todo: slit loss.
def sensfunc(spectrum1d, exptime, airmass, extinct, standard, bandwid=None, 
             bandsep=None, n_piece=3, n_iter=5, sigma_lower=None, sigma_upper=None, 
             grow=False, use_mask=False, title='sensfunc', show=conf.show, 
             save=conf.save, path=conf.path):
    """Create sensitivity function.
    
    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    exptime : str or scalar
        Exposure time. Should be either a keyword in the header (str) or exposure time 
        itself (scalar).

    airmass : str or scalar
        Airmass. Should be either a keyword in the header (str) or airmass itself 
        (scalar).

    extinct : str, `~specutils.Spectrum1D` or `None`
        Extinction curve. Should be either a file name (str) or the extinction curve 
        itself (`~specutils.Spectrum1D`) if not `None`.

    standard : str or `~specutils.Spectrum1D`
        Standard spectrum. Should be either a file name (str) in the libaray or the 
        standard spectrum itself (`~specutils.Spectrum1D`).

    bandwid, bandsep : scalar or `None`, optional
        Bandpass widths and separations in wavelength units.

    n_piece : int, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        Default is `3`.

    n_iter : int, optional
        Number of sigma slipping iterations. Must be >= `0`.
        Default is `5`.

    sigma_lower : scalar or `None`, optional
        Number of standard deviations to use as the lower bound for the clipping limit. 
        If `None` (default), `3` is used.

    sigma_upper : scalar or `None`, optional
        Number of standard deviations to use as the upper bound for the clipping limit. 
        If `None` (default), `3` is used.

    grow : float or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits (only applied along axis, if specified).

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``spectrum1d``, the masked pixels 
        are ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    sens1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Sensitivity function.
    """

    new_spectrum1d, _, flux_obs, _, mask_obs = _validateSpectrum(
        spectrum1d, 'spectrum1d', False, use_mask)

    if isinstance(exptime, str):
        exptime = spectrum1d.meta['header'][exptime]

    # ``flux`` and ``mask`` may be 2-dimensional arrays, though each one may only have 
    # one row. Use flatten to get rid of additional dimensions.
    if flux_obs.ndim > 1:
        flux_obs = flux_obs.flatten()[:flux_obs.shape[-1]]
        mask_obs = mask_obs.flatten()[:mask_obs.shape[-1]]

    flux_obs /= exptime

    wavelength_obs = deepcopy(new_spectrum1d.wavelength.value)
    bin_edges_obs = deepcopy(new_spectrum1d.bin_edges.to(u.AA).value)
    bin_width_obs = np.diff(bin_edges_obs)

    # Check whether is reversed
    if wavelength_obs[0] > wavelength_obs[-1]:
        wavelength_obs = wavelength_obs[::-1] # [Angstrom]
        bin_edges_obs = bin_edges_obs[::-1]   # [Angstrom]
        bin_width_obs = -bin_width_obs[::-1]  # [Angstrom]
        flux_obs = flux_obs[::-1]             # [counts/s]
        mask_obs = mask_obs[::-1]
        isReversed = True
    else:
        isReversed = False

    n_obs = wavelength_obs.shape[0]

    idx_obs = np.arange(n_obs)

    # Library spectrum
    if isinstance(standard, str):
        spectrum_lib, bandpass_lib = loadStandardSpectrum(standard)

    elif isinstance(standard, Spectrum1D):
        spectrum_lib = deepcopy(standard)
        bandpass_lib = None

    else:
        raise TypeError(
            '``standard`` should be either a file name of a standard spectrum or a '
            '`~specutils.Spectrum1D` object.')

    wavelength_lib = spectrum_lib.wavelength.value # [Angstrom]
    flux_lib = spectrum_lib.flux.value             # [erg/s/cm2/Angstrom]
    if bandpass_lib is None:                       # [Angstrom]
        bandpass_lib = np.diff(spectrum_lib.bin_edges).to(u.AA).value
    else:
        bandpass_lib = bandpass_lib.value

    # Default
    if (bandwid is None) | (bandsep is None):
        bandpass_arr = np.vstack(
            [wavelength_lib - bandpass_lib / 2, wavelength_lib + bandpass_lib / 2]).T
        mask = (
            (wavelength_obs[0] <= bandpass_arr[:, 0]) 
            & (bandpass_arr[:, 1] <= wavelength_obs[-1])
        )
        wavelength = wavelength_lib[mask]
        flux_bp_lib = flux_lib[mask]
        bandpass_arr = bandpass_arr[mask]

    # Custom
    else:
        coverage = wavelength_obs[-1] - wavelength_obs[0]
        n_band = coverage // (bandwid + bandsep)
        if coverage - n_band * (bandwid + bandsep) >= bandwid:
            n_band += 1
        wavelength = (
            wavelength_obs[0] + bandwid / 2 + np.arange(n_band) * (bandwid + bandsep)
        )
        bandpass_arr = np.vstack([wavelength - bandwid / 2, wavelength + bandwid / 2]).T
        flux_bp_lib = np.zeros(bandpass_arr.shape[0])

        for i, bandpass in enumerate(bandpass_arr):

            # Edge pixels
            edges_lib = interpolate.interp1d(
                x=wavelength_lib, y=flux_lib, bounds_error=False, fill_value=np.nan, 
                assume_sorted=True)(bandpass)

            # Integral over the bandpass
            mask_bp_lib = (
                (bandpass[0] < wavelength_lib) & (wavelength_lib < bandpass[1])
            )
            x = np.hstack([bandpass[0], wavelength_lib[mask_bp_lib], bandpass[1]])
            y = np.hstack([edges_lib[0], flux_lib[mask_bp_lib], edges_lib[1]])
            flux_bp_lib[i] = np.trapz(y=y, x=x) / bandwid

    flux_bp_obs = np.zeros(bandpass_arr.shape[0]) # [counts/s/Angstrom]
    mask_bp_obs = np.zeros(bandpass_arr.shape[0], dtype=bool)

    for i, bandpass in enumerate(bandpass_arr):

        # Internal pixels
        mask = (
            (bandpass[0] < bin_edges_obs[:-1]) & (bin_edges_obs[1:] < bandpass[1])
        )
        flux_bp_obs[i] = flux_obs[mask].sum()
        mask_bp_obs[i] = np.any(mask_obs[mask])

        # Edge pixels

        # ``idx_start`` labels the pixel where ``bandpass[0]`` is in
        idx_start = idx_obs[bin_edges_obs[:-1] <= bandpass[0]][-1]

        # ``idx_end`` labels the pixel where ``bandpass[1]`` is in
        idx_end = idx_obs[bin_edges_obs[1:] >= bandpass[1]][0]

        # ``bandpass[0]`` and ``bandpass[1]`` are in the same pixel
        if idx_start == idx_end:
            flux_bp_obs[i] += (
                flux_obs[idx_end] * (bandpass[1] - bandpass[0]) / bin_width_obs[idx_end]
            )
            mask_bp_obs[i] |= mask_obs[idx_end]

        # ``bandpass[0]`` and ``bandpass[1]`` are in different pixels
        else:
            flux_bp_obs[i] += (
                flux_obs[idx_start] * (bin_edges_obs[idx_start + 1] - bandpass[0]) 
                / bin_width_obs[idx_start]
            )
            flux_bp_obs[i] += (
                flux_obs[idx_end] * (bandpass[1] - bin_edges_obs[idx_end]) 
                / bin_width_obs[idx_end]
            )
            mask_bp_obs[i] |= (mask_obs[idx_start] | mask_obs[idx_end])
    
        flux_bp_obs[i] /= (bandpass[1] - bandpass[0])

    # Negative values lead to NaNs here
    sens = 2.5 * np.log10(flux_bp_obs / flux_bp_lib) # 2.5 x log10(counts / (erg / cm2))

    # Extinction curve
    if extinct is not None:

        if isinstance(airmass, str):
            airmass = spectrum1d.meta['header'][airmass]

        if isinstance(extinct, str):
            spectrum_ext = loadExtinctionCurve(extinct)

        elif isinstance(extinct, Spectrum1D):
            spectrum_ext = deepcopy(extinct)

        else:
            raise TypeError(
                '``extinct`` should be either a file name of a extinction curve or a '
                '`~specutils.Spectrum1D` object if not `None`.')

        wavelength_ext = spectrum_ext.wavelength.value
        extinction = spectrum_ext.flux.value

        # Has to be `quadratic` ???
        extinction = interpolate.interp1d(
            x=wavelength_ext, y=extinction, kind='quadratic', bounds_error=False, 
            fill_value='extrapolate', assume_sorted=True)(wavelength)

        sens += (airmass * extinction)

    mask_sens = np.isnan(sens) | mask_bp_obs

    # Fit cubic spline function
    spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
        x=wavelength, y=sens, weight=None, mask=mask_sens, order=3, n_piece=n_piece, 
        n_iter=n_iter, sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
        use_relative=False)

    sens_fit = spl(wavelength_obs)
    # Use RMS as uncertainty
    uncertainty_sens_fit = np.full(n_obs, residual[~master_mask].std(ddof=1))

    # Plot
    _validateBool(show, 'show')

    _validateString(title, 'title')

    fig_path = _validatePath(save, path, title)

    if show | save:

        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        # Spectrum
        _plotSpectrum1D(
            ax, wavelength_obs, (flux_obs / bin_width_obs), xlabel='wavelength [A]')
        # Bandpasses
        ymin, ymax = ax.get_ylim()
        height = (ymax - ymin) * 0.06
        patches = list()
        for i, bandpass in enumerate(bandpass_arr):
            width = bandpass[1] - bandpass[0]
            rectangle = Rectangle(
                (bandpass[0], (flux_bp_obs[i] - height / 2)), width, height)
            patches.append(rectangle)
        patch_collection = PatchCollection(
            patches, facecolor='None', edgecolor='r', lw=1.5, zorder=2.5)
        ax.add_collection(patch_collection)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    plotFitting(
        x=wavelength, y=sens, residual=residual, mask=master_mask, 
        x_fit=wavelength_obs, y_fit=sens_fit, threshold_lower=threshold_lower, 
        threshold_upper=threshold_upper, xlabel='wavelength [A]', 
        ylabel='sensitivity function', title=title, show=show, save=save, path=path, 
        use_relative=False)

    unit_sens_fit = new_spectrum1d.flux.unit / u.s / u.AA / spectrum_lib.flux.unit

    if isReversed:
        sens_fit = sens_fit[::-1] * unit_sens_fit
        uncertainty_sens_fit = StdDevUncertainty(uncertainty_sens_fit[::-1])
    else:
        sens_fit *= unit_sens_fit
        sens_fit_uncertainty = StdDevUncertainty(uncertainty_sens_fit)

    if 'header' in new_spectrum1d.meta:
        meta = deepcopy(new_spectrum1d.meta)
    else:
        meta = {'header': dict()}
    # Add headers here
    meta['header']['EXPTIME'] = exptime
    meta['header']['AIRMASS'] = airmass
    meta['header']['SENSFUNC'] = '{}'.format(
        Time.now().to_value('iso', subfmt='date_hm'))

    # Output
    sens1d = Spectrum1D(
        spectral_axis=new_spectrum1d.spectral_axis, flux=sens_fit, 
        uncertainty=uncertainty_sens_fit, meta=meta)

    return sens1d


def calibrate1d(spectrum1d, exptime, airmass, extinct, sens1d, use_uncertainty=False):
    """Apply a flux calibration to 1-dimensional spectra.
    
    ``spectrum1d`` and ``sens1d`` are assumed to have the same spectral axis, including 
    values, units and order (reversed or not).

    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    exptime : str or scalar
        Exposure time. Should be either a keyword in the header (str) or exposure time 
        itself (scalar).

    airmass : str or scalar
        Airmass. Should be either a keyword in the header (str) or airmass itself 
        (scalar).

    extinct : str, `~specutils.Spectrum1D` or `None`
        Extinction curve. Should be either a file name (str) or the extinction curve 
        itself (`~specutils.Spectrum1D`) if not `None`.

    sens1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Sensitivity function.

    use_uncertainty : bool, optional
        If `True`, the uncertainty array attributed to ``sens1d`` is propagated.
        Default is `False`.

    Returns
    -------
    new_spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Calibrated spectrum.
    """

    # ``flux_obs`` can be multi-dimensional array
    new_spectrum1d, _, _, _, _ = _validateSpectrum(
        spectrum1d, 'spectrum1d', False, False)

    wavelength_obs = new_spectrum1d.wavelength                     # [Angstrom]
    bin_width = np.abs(np.diff(new_spectrum1d.bin_edges.to(u.AA))) # [Angstrom]

    if isinstance(exptime, str):
        exptime = spectrum1d.meta['header'][exptime] * u.s

    flux_obs = new_spectrum1d.flux.value / (exptime * bin_width) # [counts/s/Angstrom]

    if new_spectrum1d.uncertainty is not None:
        uncertainty_obs = StdDevUncertainty(
            new_spectrum1d.uncertainty.array / (exptime.value * bin_width.value))
    else:
        uncertainty_obs = None

    new_spectrum1d = Spectrum1D(
        spectral_axis=wavelength_obs, flux=flux_obs, uncertainty=uncertainty_obs, 
        mask=new_spectrum1d.mask, meta=new_spectrum1d.meta)

    _validateBool(use_uncertainty, 'use_uncertainty')

    new_sens1d, _, sens, uncertainty_sens, _ = _validateSpectrum(
        sens1d, 'sens1d', use_uncertainty, False)

    wavelength_sens = new_sens1d.wavelength.value # [Angstrom]

    # ``wavelength_sens`` is dimensionless
    if not (wavelength_obs.value == wavelength_sens).all():
        raise ValueError(
            'The spectral axes of ``spectrum1d`` and ``sens1d`` should be the same.')

    # ``sens``, ``uncertainty_sens`` should be 1-dimensional arrays. 
    # Use flatten to get rid of additional dimensions.
    if sens.ndim > 1:
        sens = sens.flatten()[:sens.shape[-1]]
        uncertainty_sens = uncertainty_sens.flatten()[:uncertainty_sens.shape[-1]]

    # Extinction curve
    if extinct is not None:

        if isinstance(airmass, str):
            airmass = spectrum1d.meta['header'][airmass]

        if isinstance(extinct, str):
            spectrum_ext = loadExtinctionCurve(extinct)

        elif isinstance(extinct, Spectrum1D):
            spectrum_ext = deepcopy(extinct)

        else:
            raise TypeError(
                '``extinct`` should be either a file name of a extinction curve or a '
                '`~specutils.Spectrum1D` object if not `None`.')

        wavelength_ext = spectrum_ext.wavelength.value
        extinction = spectrum_ext.flux.value

        extinction = interpolate.interp1d(
            x=wavelength_ext, y=extinction, kind='quadratic', bounds_error=False, 
            fill_value='extrapolate', assume_sorted=True)(wavelength_sens)

        sens -= (airmass * extinction)

    sens = 10**(0.4 * sens)                 # [counts / (erg / cm2)]
    uncertainty_sens *= 0.4 * np.log(10) * sens  # [counts / (erg / cm2)]

    new_sens1d = Spectrum1D(
        spectral_axis=(wavelength_sens * u.AA), flux=(sens * new_sens1d.flux.unit), 
        uncertainty=StdDevUncertainty(uncertainty_sens), meta=new_sens1d.meta)

    # Calibrate
    calibrated_spectrum1d = new_spectrum1d / new_sens1d
    
    # Output
    if new_spectrum1d.uncertainty is None:
        calibrated_spectrum1d.uncertainty = None

    if 'header' in new_spectrum1d.meta:
        meta = deepcopy(new_spectrum1d.meta)
    else:
        meta = {'header': dict()}
    # Add headers here
    meta['header']['EXPTIME'] = exptime.value
    meta['header']['AIRMASS'] = airmass
    meta['header']['CALIBRAT'] = '{} Calibrated'.format(
        Time.now().to_value('iso', subfmt='date_hm'))

    calibrated_spectrum1d.meta = meta

    return calibrated_spectrum1d
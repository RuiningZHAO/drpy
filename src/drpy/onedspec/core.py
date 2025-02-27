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
from astropy.table import QTable
from astropy.nddata import StdDevUncertainty
# specutils
from specutils import Spectrum1D
# drpy
from drpy import conf
from drpy.modeling import Poly1D, Spline1D
from drpy.plotting import plotFitting, _plotSpectrum1D
from drpy.validate import (_validateBool, _validateString, _validateInteger, 
                           _validateSpectrum, _validatePath)

from .center import _center1D_Gaussian, _refinePeakBases, _refinePeaks
from .io import loadStandardSpectrum, loadExtinctionCurve#, loadSpectrum1D

__all__ = ['dispcor', 'sensfunc', 'calibrate1d']


def dispcor(spectrum1d, reverse, reference, n_sub=20, refit=True, prominence=1e-3, 
            degree=1, maxiters=5, sigma_lower=None, sigma_upper=None, grow=False, 
            use_mask=False, title='dispcor', show=conf.fig_show, save=conf.fig_save, 
            path=conf.fig_path, **kwargs):
    """Dispersion correction.

    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    reverse : bool
        Reverse the input spectrum or not. Set to `True` if the spectral axis of the 
        input spectrum is in reverse order.

    reference : `~specutils.Spectrum1D` or str
        File name of the reference spectrum.

    n_sub : int
        A pixel is divided into ``n_sub`` subpixels before cross-correlation.
        Default is `20`.

    refit : bool, optional
        Refit the dispersion solution or not. If `True`, lines in the input spectrum 
        are recentered and then used to fit the dispersion solution.

    degree : int, optional
        Degree of the fitting polynomial.
        Default is `3`.

    maxiters : int, optional
        Maximum number of sigma-clipping iterations to perform. If convergence is 
        achieved prior to ``maxiters`` iterations, the clipping iterations will stop. 
        Must be >= `0`. 
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

    flux /= np.nanmax(flux)
    
    _validateBool(reverse, 'reverse')

    if reverse:
        flux = flux[::-1]

    # Parse reference spectrum
    if isinstance(reference, Spectrum1D):
        spectrum1d_ref = deepcopy(reference)
    else:
        spectrum1d_ref = Spectrum1D.read(reference)
    spectral_axis_ref = spectrum1d_ref.spectral_axis.value
    unit_spectral_axis = spectrum1d_ref.spectral_axis.unit
    flux_ref = spectrum1d_ref.flux.value
    flux_ref /= np.nanmax(flux_ref)

    n_pix = spectral_axis_ref.shape[0]

    index_ref = np.arange(n_pix)

    n_sub = _validateInteger(n_sub, 'n_sub', (1, None), (True, None))

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
    delta = index_ref_dense_ext[index_max]

    _validateBool(refit, 'refit')

    if refit:

        # Find peaks in the reference spectrum
        peaks, properties = signal.find_peaks(
            x=flux_ref, prominence=prominence, **kwargs)
        # Refine peaks in the reference spectrum
        heights = flux_ref[peaks]
        refined_left_bases, refined_right_bases = _refinePeakBases(
            peaks=peaks, left_bases=properties['left_bases'], 
            right_bases=properties['right_bases'], n_peak=peaks.shape[0], copy=True)
        refined_peaks_ref, refined_index_ref = _refinePeaks(
            flux_ref, peaks, heights, refined_left_bases, refined_right_bases, 1)
    
        # Reidentify in the input spectrum
        shifted_peaks = refined_peaks_ref - delta
        heights = heights[refined_index_ref]
        shifted_left_bases = refined_left_bases[refined_index_ref] - delta
        shifted_right_bases = refined_right_bases[refined_index_ref] - delta
        refined_peaks, refined_index = _refinePeaks(
            flux, shifted_peaks, heights, shifted_left_bases, shifted_right_bases, 1)

        # Fit polynomial
        delta_peaks = refined_peaks_ref[refined_index] - refined_peaks
        spl, residual, threshold_lower, threshold_upper, master_mask = Poly1D(
            x=refined_peaks, y=delta_peaks, deg=degree, maxiters=maxiters, 
            sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)
        delta = spl(index)

        rms = np.sqrt((residual[~master_mask]**2).sum() / (~master_mask).sum())

        # print(f'dispersion solution rms = {rms:.3}')

        plotFitting(
            x=refined_peaks, y=delta_peaks, m=master_mask, x_fit=index, y_fit=delta, 
            r=residual, threshold_lower=threshold_lower, 
            threshold_upper=threshold_upper, xlabel='dispersion axis [px]', 
            ylabel='difference [px]', title='dispcor', show=show, save=save, 
            path=path, use_relative=False)

    # !!! The extrapolated wavelengths are not reliable !!!
    spectral_axis = interpolate.interp1d(
        index_ref, spectral_axis_ref, bounds_error=False, fill_value='extrapolate', 
        assume_sorted=True)(index + delta)

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'dispcor':
            title = f'{title} dispcor'

        # Peak detection plot

        # Split into subplots
        n_subplot = 2

        fig, ax = plt.subplots(n_subplot, 1, dpi=100)

        length = n_pix // n_subplot + 1
        for i in range(n_subplot):

            idx_start, idx_end = i * length, (i + 1) * length
            idx_peak = np.where(
                (idx_start <= refined_peaks_ref[refined_index]) 
                & (refined_peaks_ref[refined_index] < idx_end))[0]

            ax[i].step(
                index_ref[idx_start:idx_end], flux_ref[idx_start:idx_end], color='k', 
                ls='-', where='mid')
            for idx in idx_peak:
                ymin = heights[refined_index][idx] * 1.2
                ymax = heights[refined_index][idx] * 1.5
                ax[i].plot(
                    [refined_peaks_ref[refined_index][idx], 
                     refined_peaks_ref[refined_index][idx]], [ymin, ymax], 'r-', 
                    lw=1.5)

            # Settings
            ax[i].grid(axis='both', color='0.95', zorder=-1)
            ax[i].set_xlim(idx_start, idx_end)
            ax[i].set_yscale('log')
            ax[i].tick_params(
                which='major', direction='in', top=True, right=True, length=5, 
                width=1.5, labelsize=12)
            ax[i].tick_params(
                which='minor', direction='in', top=True, right=True, length=3, 
                width=1.5, labelsize=12)
            ax[i].set_ylabel('normalized intensity', fontsize=16)
        ax[-1].set_xlabel('dispersion axis [px]', fontsize=16)
        ax[0].set_title(f'{title} peak detection', fontsize=16)
        fig.align_ylabels()
        fig.set_figheight(fig.get_figwidth() * n_subplot / 2)
        fig.tight_layout()

        # Save
        if save:
            fig_path = _validatePath(path, f'{title} peak detection')
            plt.savefig(fig_path, dpi=100)
        
        if show:
            plt.show()

        plt.close()
        
        # Comparison
        xlabel = f'spectral axis [{unit_spectral_axis.to_string()}]'

        fig, ax = plt.subplots(1, 1, dpi=100)

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
        fig.set_figheight(fig.get_figwidth() * 2 / 3)
        fig.tight_layout()
        
        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    if reverse:
        spectral_axis = spectral_axis[::-1]

    spectral_axis *= unit_spectral_axis

    if 'header' in new_spectrum1d.meta:
        meta = new_spectrum1d.meta.copy()

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


# Old version
def _dispcor(spectrum1d, reverse, reference, n_piece=3, refit=True, maxiters=5, 
            sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
            title='dispcor', show=conf.fig_show, save=conf.fig_save, 
            path=conf.fig_path):
    """Dispersion correction.

    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    reverse : bool
        Reverse the input spectrum or not. Set to `True` if the spectral axis of the 
        input spectrum is in reverse order.

    reference : str
        File name of the reference spectrum.

    n_piece : int, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        Default is `3`.

    refit : bool, optional
        Refit the dispersion solution or not. If `True`, lines in the input spectrum 
        are recentered and then used to fit the dispersion solution.

    maxiters : int, optional
        Maximum number of sigma-clipping iterations to perform. If convergence is 
        achieved prior to ``maxiters`` iterations, the clipping iterations will stop. 
        Must be >= `0`. 
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

    spectrum1d_ref = loadSpectrum1D(reference, ext='spec')
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
        peak_tbl = QTable.read(reference, format='fits', hdu='peak')

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
            x=refined_peaks, y=spectral_axis_peaks, order=3, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)
        spectral_axis = spl(index)

        rms = np.sqrt((residual[~master_mask]**2).sum() / (~master_mask).sum())

        print(f'dispersion solution rms = {rms:.3}')

        plotFitting(
            x=spectral_axis_peaks, y=refined_peaks, m=master_mask, x_fit=spectral_axis, 
            y_fit=index, r=residual, threshold_lower=threshold_lower, 
            threshold_upper=threshold_upper, 
            xlabel=f'spectral axis [{unit_spectral_axis.to_string()}]', 
            ylabel='dispersion axis [px]', title='dispersion solution', show=show, 
            save=save, path=path, use_relative=False)

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'dispcor':
            title = f'{title} dispcor'

        xlabel = f'spectral axis [{unit_spectral_axis.to_string()}]'

        fig, ax = plt.subplots(1, 1, dpi=100)

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
        fig.set_figheight(0.5 * fig.get_figwidth())
        fig.tight_layout()
        
        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    if reverse:
        spectral_axis = spectral_axis[::-1]

    spectral_axis *= unit_spectral_axis

    if 'header' in new_spectrum1d.meta:
        meta = new_spectrum1d.meta.copy()

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


def sensfunc(spectrum1d, exptime, airmass, extinct, standard, bandwid=None, 
             bandsep=None, n_piece=3, maxiters=5, sigma_lower=None, sigma_upper=None, 
             grow=False, use_mask=False, title='sensfunc', show=conf.fig_show, 
             save=conf.fig_save, path=conf.fig_path):
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

    maxiters : int, optional
        Maximum number of sigma-clipping iterations to perform. If convergence is 
        achieved prior to ``maxiters`` iterations, the clipping iterations will stop. 
        Must be >= `0`. 
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

    new_spectrum1d, _, flux_obs, uncertainty_obs, mask_obs = _validateSpectrum(
        spectrum1d, 'spectrum1d', False, use_mask)

    if isinstance(exptime, str):
        exptime = spectrum1d.meta['header'][exptime]

    # ``flux`` and ``mask`` may be 2-dimensional arrays, though each one may only have 
    # one row. Use flatten to get rid of additional dimensions.
    if flux_obs.ndim > 1:
        flux_obs = flux_obs.flatten()[:flux_obs.shape[-1]]
        mask_obs = mask_obs.flatten()[:mask_obs.shape[-1]]

    flux_obs /= exptime

    wavelength_obs = new_spectrum1d.wavelength.value.copy()
    bin_edges_obs = new_spectrum1d.bin_edges.to(u.AA).value.copy()
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
        x=wavelength, y=sens, m=mask_sens, order=3, n_piece=n_piece, maxiters=maxiters, 
        sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, use_relative=False)

    sens_fit = spl(wavelength_obs)
    # Use RMS as uncertainty
    uncertainty_sens_fit = np.full(n_obs, residual[~master_mask].std(ddof=1))
    
    out_of_range = (
        (wavelength_obs < wavelength[~master_mask][0]) | 
        (wavelength_obs > wavelength[~master_mask][-1])
    )
    sens_fit[out_of_range] = np.nan
    uncertainty_sens_fit[out_of_range] = np.nan

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')

        fig, ax = plt.subplots(1, 1, dpi=100)

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

        # Settings
        ax.set_title(title, fontsize=16)
        fig.set_figheight(fig.get_figwidth() * 2 / 3)
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    # Sensitivity function fitting
    plotFitting(
        x=wavelength, y=sens, m=master_mask, x_fit=wavelength_obs, y_fit=sens_fit, 
        r=residual, threshold_lower=threshold_lower, threshold_upper=threshold_upper, 
        xlabel='wavelength [A]', ylabel='sensitivity function', title=title, show=show, 
        save=save, path=path, use_relative=False)

    unit_sens_fit = new_spectrum1d.flux.unit / u.s / u.AA / spectrum_lib.flux.unit

    if isReversed:
        sens_fit = sens_fit[::-1] * unit_sens_fit
        uncertainty_sens_fit = StdDevUncertainty(uncertainty_sens_fit[::-1])
    else:
        sens_fit *= unit_sens_fit
        uncertainty_sens_fit = StdDevUncertainty(uncertainty_sens_fit)

    if 'header' in new_spectrum1d.meta:
        meta = new_spectrum1d.meta.copy()
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

    return sens1d, spl


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

    flux_obs = new_spectrum1d.flux / (exptime * bin_width) # [counts/s/Angstrom]

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

    sens = 10**(0.4 * sens) # [counts / (erg / cm2)]
    uncertainty_sens *= 0.4 * np.log(10) * sens # [counts / (erg / cm2)]

    new_sens1d = Spectrum1D(
        spectral_axis=(wavelength_sens * u.AA), flux=(sens * new_sens1d.flux.unit), 
        uncertainty=StdDevUncertainty(uncertainty_sens), meta=new_sens1d.meta)

    # Calibrate
    calibrated_spectrum1d = new_spectrum1d / new_sens1d
    
    # Output
    if new_spectrum1d.uncertainty is None:
        calibrated_spectrum1d.uncertainty = None

    if 'header' in new_spectrum1d.meta:
        meta = new_spectrum1d.meta.copy()
    else:
        meta = {'header': dict()}
    # Add headers here
    meta['header']['EXPTIME'] = exptime.value
    meta['header']['AIRMASS'] = airmass
    meta['header']['CALIBRAT'] = '{} Calibrated'.format(
        Time.now().to_value('iso', subfmt='date_hm'))

    calibrated_spectrum1d.meta = meta

    return calibrated_spectrum1d
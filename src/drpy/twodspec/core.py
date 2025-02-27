from copy import deepcopy
import warnings

# NumPy
import numpy as np
# SciPy
from scipy import interpolate, signal, ndimage
from scipy.optimize import OptimizeWarning
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc.utils.slices import slice_from_string
# specutils
from specutils import Spectrum1D
# drpy
from drpy import conf
from drpy.onedspec.center import _center1D_Gaussian, _refinePeakBases, _refinePeaks
from drpy.modeling import Poly1D, Spline1D, Spline2D
from drpy.plotting import plotFitting, _plotFitting, plot2d, _plot2d
from drpy.validate import (_validateBool, _validateString, _validateRange, 
                           _validateInteger, _validate1DArray, _validateNDArray, 
                           _validateCCD, _validateBins, _validateAperture, 
                           _validatePath)
from drpy.decorate import filterWarning

sigma_clip = filterWarning('ignore', AstropyUserWarning)(sigma_clip)

# Set plot parameters
plt.rcParams['figure.figsize'] = [conf.fig_width, conf.fig_width]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['trace', 'background', 'profile', 'extract']


# todo: doc
@filterWarning('error', OptimizeWarning)
def _trace(mode, method, count_bin, mask_bin, loc_bin, n_bin, k, interval, fwhm, 
           idx_row, idx_col, order, n_piece, maxiters, sigma_lower, sigma_upper, grow):
    """
    """

    # Maximum location in a given interval, which ideally should contain only one trace.
    slice_interval = slice_from_string(interval)
    slice_interval_start = (
        slice_interval[0].start if slice_interval[0].start is not None else 0
    )
    idx_src_ref = slice_interval_start + np.ma.argmax(
        np.ma.array(count_bin[k], mask=mask_bin[k])[slice_interval], fill_value=-np.inf)

    # Peak range
    idx_min_ref = np.max([int(idx_src_ref - 0.5 * fwhm), 0])
    idx_max_ref = np.min([int(idx_src_ref + 0.5 * fwhm), idx_row[-1]])

    x_ref = idx_min_ref + np.arange(idx_max_ref + 1 - idx_min_ref)
    y_ref = deepcopy(count_bin[k][idx_min_ref:(idx_max_ref + 1)])
    m_ref = mask_bin[k][idx_min_ref:(idx_max_ref + 1)]

    # Normalize
    y_min, y_max = y_ref[~m_ref].min(), count_bin[k][idx_src_ref]
    if y_min < 0:
        y_ref -= y_min
        y_max -= y_min
    if y_max != 0:
        y_ref /= y_max
    else:
        # To avoid warning caused by dividing zero.
        y_ref = np.zeros_like(x_ref) + np.nan

    # Initial guess for Gaussian fitting   
    initial_guess = (1, idx_src_ref, fwhm)

    # Gaussian fitting
    try:
        center_ref = _center1D_Gaussian(x_ref[~m_ref], y_ref[~m_ref], initial_guess, 0)

    except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
        raise RuntimeError('No trace found in the given interval.')

    if mode == 'center':

        return center_ref

    else:

        # interval_width = count_bin[k][slice_interval].shape[0]

        refined_trace = np.full(n_bin, np.nan)
        refined_trace[k] = center_ref

        # Trace the two directions in a common loop
        index_arr = np.hstack(
            [np.arange(k + 1, n_bin), np.arange(k - 1, -1, -1)])

        for i in index_arr:

            # Peak range
            if i >= k:
                idx_ref = np.where(~np.isnan(refined_trace))[0].max()
            elif i < k:
                idx_ref = np.where(~np.isnan(refined_trace))[0].min()
            interval_min = np.max(
                [int(refined_trace[idx_ref] - 0.5 * fwhm), 0])
            interval_max = np.min(
                [int(refined_trace[idx_ref] + 0.5 * fwhm), idx_row[-1]])

            # Peak center
            idx_src_bin = interval_min + np.ma.argmax(
                np.ma.array(count_bin[i], mask=mask_bin[i])[interval_min:interval_max], 
                fill_value=-np.inf)

            # Centroiding range
            idx_min_bin = np.max([int(idx_src_bin - 0.5 * fwhm), 0])
            idx_max_bin = np.min([int(idx_src_bin + 0.5 * fwhm), idx_row[-1]])

            x_bin = idx_min_bin + np.arange(idx_max_bin + 1 - idx_min_bin)
            y_bin = deepcopy(count_bin[i][idx_min_bin:(idx_max_bin + 1)])
            m_bin = mask_bin[i][idx_min_bin:(idx_max_bin + 1)]

            if np.all(m_bin):
                continue
            else:
                y_min = y_bin[~m_bin].min()

            if method == 'gaussian':

                # Normalize
                y_max = count_bin[i][idx_src_bin]
                if y_min < 0:
                    y_bin -= y_min
                    y_max -= y_min
                if y_max != 0:
                    y_bin /= y_max
                else:
                    # To avoid warning caused by dividing zero.
                    y_bin = np.zeros_like(x_bin) + np.nan

                # Initial guess
                initial_guess = (1, idx_src_bin, fwhm)

                # Gaussian fitting
                try:
                    refined_trace[i] = _center1D_Gaussian(
                        x_bin[~m_bin], y_bin[~m_bin], initial_guess, 0)

                # raise exception here
                except (RuntimeError, TypeError, OptimizeWarning, ValueError):
                    continue

            else:

                if y_min < 0:
                    y_bin -= y_min

                if y_bin.sum() != 0:
                    com = (x_bin * y_bin).sum() / y_bin.sum()

                    if x_bin[0] < com < x_bin[-1]:
                        refined_trace[i] = com

        mask = np.isnan(refined_trace)

        # Spline fitting
        spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
            x=loc_bin, y=refined_trace, m=mask, order=order, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
            grow=grow, use_relative=False)

        fitted_trace = spl(idx_col)

        return (center_ref, fitted_trace, refined_trace, residual, master_mask, 
                threshold_lower, threshold_upper)


def trace(ccd, dispersion_axis, fwhm, mode, method, n_med=3, reference_bin=None, 
          interval='[:]', order=3, n_piece=3, maxiters=5, sigma_lower=None, 
          sigma_upper=None, grow=False, negative=False, use_mask=False, title='trace', 
          show=conf.fig_show, save=conf.fig_save, path=conf.fig_path):
    """Trace on the 2-dimensional spectrum.
    
    First the spatial profiles are binned by taking median along the dispersion axis. 
    Then the center of the specified feature (that is the strongest one in the 
    interval) in the reference bin is determined by fitting a Gaussian profile. [...]
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    dispersion_axis : str
        `col` or `row`. If `col`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.

    fwhm : scalar
        Estimated full width at half maximum of the peak to be traced. (Rough 
        estimation is enough.)

    mode : str
        `center` or `trace`. If `center`, a straight line going through the peak center 
        of the middle bin is returned.
    
    method : str
        Centroiding method used in tracing. `gaussian` or `com`, and
        - if `gaussian`, peak center in each bin is determined using Gaussian fitting.
        This method is easy to fail when the data is noisy (for example, near the
        edges). Peak centers of these bins appear as NaNs in the trace fitting, so that
        they do not affect the result much as long as there are enough good peak
        centers.
        - if `com`, center of mass in each bin is determined. This method always
        provides peak center even when the data is quite noisy. The mask is not applied
        as well. A proper sigma-clipping is needed to avoid impact of bad peak centers
        on the trace fitting.
    
    n_med : int, optional
        Number of spatial profiles to median. Must be >= `3`. Large number for faint 
        source.
        Default is `3`.
    
    reference_bin : int or `None`, optional
        Index of the reference bin.
        If `None`, the reference bin is the middle bin.
    
    interval : str, optional
        Spatial interval the specified feature lies in. The brightest feature in the 
        interval is traced. (Use this when the target is not the brightest one in the 
        frame)
        Default is `[:]`.
    
    order : int, optional
        Degree of the spline. Must be `5` >= ``order`` >= `1`.
        Default is `3`, a cubic spline.
    
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
    
    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits.

    negative : bool, optional
        The spectrum is negative or not. If `True`, the negative frame (the input frame 
        multiplied by `-1`) is used in traceing.
        Default is `False`.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    trace1d : 
        Fitted trace.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    fwhm = np.abs(fwhm)

    _validateBool(negative, 'negative')

    _validateBool(use_mask, 'use_mask')

    if dispersion_axis == 'row':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    if negative:
        data_arr *= -1

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Split into bins along dispersion axis
    _validateInteger(n_med, 'n_med', (3, None), (True, None))
    bin_edges, loc_bin, n_bin = _validateBins(n_med, n_col, isWidth=True)

    # If `None`, the reference bin will be the middle bin when the total number is odd 
    # while the first bin of the second half when the total number is even.
    if reference_bin is not None:
        _validateInteger(reference_bin, 'reference_bin', (0, (n_bin - 1)), (True, True))
    else:
        reference_bin = n_bin // 2

    count_bin = np.zeros([n_bin, n_row])
    mask_bin = np.zeros([n_bin, n_row], dtype=bool)

    for i in range(n_bin):

        bin_edge = bin_edges[i]

        # Bad pixels (NaNs or infs) in the original frame (if any) may lead to unmasked
        # elements in ``count_bin[i]`` and may cause an error in the Gaussian fitting
        # below.
        count_bin[i] = np.nanmedian(data_arr[:, bin_edge[0]:bin_edge[1]], axis=1)
        mask_bin[i] = np.all(mask_arr[:, bin_edge[0]:bin_edge[1]], axis=1)

        # # Sometimes the median profile is still noisy
        # if mask_bin[i].sum() > 0:
        #     count_bin[i] = interpolate.interp1d(
        #         idx_row[~mask_bin[i]], count_ref[~mask_bin[i]], bounds_error=False, 
        #         fill_value='extrapolate')(idx_row)
        # count_bin[i] = ndimage.gaussian_filter(count_bin[i], fwhm)

    _validateString(mode, 'mode', ['center', 'trace'])
    _validateString(method, 'method', ['gaussian', 'com'])

    if mode == 'center':

        center_ref = _trace(
            mode=mode, method=method, count_bin=count_bin, mask_bin=mask_bin, 
            loc_bin=loc_bin, n_bin=n_bin, k=reference_bin, interval=interval, 
            fwhm=fwhm, idx_row=idx_row, idx_col=idx_col, order=order, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
            grow=grow)
        fitted_trace = np.full(n_col, center_ref)
        
    else:
        (center_ref, fitted_trace, refined_trace, residual, 
         master_mask, threshold_lower, threshold_upper) = _trace(
            mode=mode, method=method, count_bin=count_bin, mask_bin=mask_bin, 
            loc_bin=loc_bin, n_bin=n_bin, k=reference_bin, interval=interval, 
            fwhm=fwhm, idx_row=idx_row, idx_col=idx_col, order=order, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
            grow=grow)

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'trace':
            title = f'{title} trace'

        # Trace fitting plot
        if mode == 'trace':
            plotFitting(
                x=loc_bin, y=refined_trace, m=master_mask, x_fit=idx_col, 
                y_fit=fitted_trace, r=residual, threshold_lower=threshold_lower, 
                threshold_upper=threshold_upper, xlabel='dispersion axis [px]', 
                ylabel='spatial axis [px]', title=title, show=show, save=save, 
                path=path, use_relative=False)

        # Trace image
        height_ratios = (1 / 4.5, n_col / n_row)
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, dpi=100)

        # Subplot 1
        ax[0].step(idx_row, count_bin[reference_bin], 'k-', lw=1.5, where='mid')
        ax[0].axvline(x=center_ref, color='r', ls='--', lw=1.5)

        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
        # ax[0].set_yscale('log')
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_ylabel('pixel value', fontsize=16)
        ax[0].set_title(title, fontsize=16)

        # Subplot 2
        if dispersion_axis == 'row':
            xlabel, ylabel = 'row', 'column'
        else:
            xlabel, ylabel = 'column', 'row'
        _plot2d(
            ax=ax[1], ccd=data_arr.T, cmap='Greys_r', contrast=0.25, cbar=False, 
            xlabel=xlabel, ylabel=ylabel, aspect='auto')
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        ax[1].plot(fitted_trace, idx_col, 'r--', lw=1.5)

        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.set_figheight(fig.get_figwidth() * np.sum(height_ratios))
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    header = nccd.header.copy()
    header['TRINTERV'] = interval
    header['TRCENTER'] = center_ref
    header['TRACE'] = '{} Trace ({}, n_med = {}, n_piece = {})'.format(
        Time.now().to_value('iso', subfmt='date_hm'), method, n_med, n_piece)

    meta = {'header': header}

    # No uncertainty or mask frame
    trace1d = Spectrum1D(flux=(fitted_trace * u.pixel), meta=meta)

    return trace1d


def background(ccd, dispersion_axis, trace1d=None, location=75, aper_width=50, 
               degree=0, maxiters=5, sigma_lower=None, sigma_upper=None, grow=False, 
               use_uncertainty=False, use_mask=False, title='background', 
               show=conf.fig_show, save=conf.fig_save, path=conf.fig_path):
    """Model background.
    
    Sky background of the input frame is modeled col by col (or row by row, depending 
    on the ``dispersion_axis``) through polynomial fittings.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    dispersion_axis : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.

    trace1d : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray` or `None`, optional
        Input trace.

    location : scalar or 2-tuple, optional
        Location of the background apertures. If ``trace1d`` is `None`, ``location`` is 
        taken as absolute location along the slit, otherwise it is taken as the 
        distances from the input trace.

    aper_width : scalar or tuple, optional
        Aperture widths of the background apertures.

    degree : int, optional
        Degree of the fitting polynomial.
        Default is `0`.
    
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
    
    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits.

    use_uncertainty : bool, optional
        If `True` and an uncertainty array is attributed to ``ccd``, the uncertainties 
        are used as weights in the fitting. Note that weighted fitting is biased 
        towards low values.
        Default is `False`.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Modeled background.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    _validateBool(use_uncertainty, 'use_uncertainty')

    _validateBool(use_mask, 'use_mask')

    if dispersion_axis == 'row':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', use_uncertainty, use_mask, False, asWeight=True)

    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', use_uncertainty, use_mask, True, asWeight=True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Weights (inverse variance weighted)
    wght_arr = 1 / uncertainty_arr

    # Assume that unit of the trace is [pixel]. No NaN is assumed.
    if trace1d is None:
        trace1d = np.zeros(n_col)
    elif isinstance(trace1d, Spectrum1D):
        trace1d = trace1d.flux.value
    else:
        trace1d = _validate1DArray(trace1d, 'trace1d', n_col, True)

    if np.ndim(location) == 0:
        location = np.array([location])
    else:
        location = _validateNDArray(location, 'location', 1)

    n_aper = location.shape[0]

    aper_width = _validate1DArray(aper_width, 'aper_width', n_aper, True)

    idx_min_arr = np.zeros((n_aper, n_col))
    idx_max_arr = np.zeros((n_aper, n_col))
    mask_background = np.zeros((n_aper, n_col, n_row), dtype=bool)
    for i in range(n_aper):

        _validateRange(aper_width[i], f'aper_width[{i}]', (1, None), (True, None))

        # Usually the background apertures are separated (though they can be set 
        # overlapped). ``location`` controls separations on either side of the trace.
        idx_min_arr[i] = (trace1d + location[i] - aper_width[i] / 2)
        idx_max_arr[i] = (trace1d + location[i] + aper_width[i] / 2)

        idx_row_arr = np.tile(idx_row, (n_col, 1))
        mask_background[i] = (
            (idx_min_arr[i, np.newaxis].T <= idx_row_arr) 
            & (idx_row_arr <= idx_max_arr[i, np.newaxis].T)
        )

    mask_background = np.any(mask_background, axis=0)

    idx_bkgd_min = idx_min_arr.min()
    idx_bkgd_max = idx_max_arr.max()
    if (idx_bkgd_min < -0.5) | (idx_bkgd_max > (n_row - 0.5)):
        warnings.warn('Background index out of range.', RuntimeWarning)

    if np.any(mask_background.sum(axis=1) == 0):
        raise RunTimeError('Background index out of range.')

    # Background fitting
    bkgd_arr = np.zeros_like(data_arr)
    rsdl_arr = np.zeros_like(data_arr)
    threshold_lower = [None] * n_col
    threshold_upper = [None] * n_col
    for i in range(n_col):

        mask_bkgd = mask_background[i]

        p, rsdl_arr[mask_bkgd, i], threshold_lower[i], threshold_upper[i], \
        mask_arr[mask_bkgd, i] = Poly1D(
            x=idx_row[mask_bkgd], y=data_arr[mask_bkgd, i], w=wght_arr[mask_bkgd, i], 
            m=mask_arr[mask_bkgd, i], deg=degree, maxiters=maxiters, 
            sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)

        bkgd_arr[:, i] = p(idx_row)
        rsdl_arr[~mask_bkgd, i] = data_arr[~mask_bkgd, i] - bkgd_arr[~mask_bkgd, i]

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'background':
            title = f'{title} background'
        
        # Background image
        height_ratios = (1 / 4.5, n_col / n_row)
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, dpi=100)

        # Subplot 1
        ax[0].step(idx_row, np.nanmedian(data_arr, axis=1), 'k-', lw=1.5, where='mid')
        for i in range(n_aper):
            ax[0].axvline(x=idx_min_arr[i].mean(), color=f'C{i}', ls='--', lw=1.5)
            ax[0].axvline(x=idx_max_arr[i].mean(), color=f'C{i}', ls='--', lw=1.5)

        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_ylabel('pixel value', fontsize=16)
        ax[0].set_title(title, fontsize=16)

        # Subplot 2
        if dispersion_axis == 'row':
            xlabel, ylabel = 'row', 'column'
        else:
            xlabel, ylabel = 'column', 'row'
        _plot2d(
            ax=ax[1], ccd=data_arr.T, cmap='Greys_r', contrast=0.25, cbar=False, 
            xlabel=xlabel, ylabel=ylabel, aspect='auto')
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        for i in range(n_aper):
            ax[1].plot(idx_min_arr[i], idx_col, '--', color=f'C{i}', lw=1.5)
            ax[1].plot(idx_max_arr[i], idx_col, '--', color=f'C{i}', lw=1.5)

        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.set_figheight(fig.get_figwidth() * np.sum(height_ratios))
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, f'{title} aperture')
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    # Background fitting plot
    if save:

        idx_bkgd_range = idx_bkgd_max - idx_bkgd_min
        mask_plot = (
            ((idx_bkgd_min - 0.1 * idx_bkgd_range) <= idx_row) 
            & (idx_row <= (idx_bkgd_max + 0.1 * idx_bkgd_range))
        )

        fig_path = _validatePath(path, f'{title} fitting', '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in idx_col[::10]:

                fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                fig.subplots_adjust(hspace=0)

                _plotFitting(
                    ax=ax, x=idx_row, y=data_arr[:, i], m=mask_arr[:, i], 
                    x_fit=idx_row, y_fit=bkgd_arr[:, i], r=rsdl_arr[:, i], 
                    threshold_lower=threshold_lower[i], 
                    threshold_upper=threshold_upper[i], xlabel='spatial axis [px]', 
                    ylabel='pixel value', use_relative=False)

                for j in range(n_aper):
                    ax[0].axvline(x=idx_min_arr[j, i], color=f'C{i}', ls='--', lw=1.5)
                    ax[0].axvline(x=idx_max_arr[j, i], color=f'C{i}', ls='--', lw=1.5)

                # Settings
                ax[0].set_xlim(
                    idx_bkgd_min - 0.1 * idx_bkgd_range, 
                    idx_bkgd_max + 0.1 * idx_bkgd_range)
                ax[0].set_title(f'background at column {i}', fontsize=16)

                for j in range(n_aper):
                    ax[1].axvline(x=idx_min_arr[j, i], color=f'C{i}', ls='--', lw=1.5)
                    ax[1].axvline(x=idx_max_arr[j, i], color=f'C{i}', ls='--', lw=1.5)

                # Settings
                ax[1].set_xlim(
                    idx_bkgd_min - 0.1 * idx_bkgd_range, 
                    idx_bkgd_max + 0.1 * idx_bkgd_range)

                fig.align_ylabels()
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()

    # Background frame
    if dispersion_axis == 'row':

        nccd.data = bkgd_arr.copy()
        nccd.uncertainty = None
        if nccd.mask is not None:
            nccd.mask = mask_arr.copy()

    else:

        nccd.data = bkgd_arr.T
        nccd.uncertainty = None
        if nccd.mask is not None:
            nccd.mask = mask_arr.T

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKBACKGR'] = '{} Background.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return nccd


def profile(ccd, dispersion_axis, trace1d, profile_width, window_length, polyorder=3, 
            deriv=0, delta=1.0, title='profile', show=conf.fig_show, 
            save=conf.fig_save, path=conf.fig_path):
    """Build an effective spatial profile along slit. Usually used by the optimal 
    extraction algorithm.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame. Should be background subtracted.

    dispersion_axis : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.

    trace1d : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray`
        Input trace.

    profile_width : scalar or 2-tuple
        Profile width. Usually large widths are preferred. Pixels outside the profile 
        will be set to `0`.
    
    window_length : int
        The length of the filter window (i.e., the number of coefficients). Must be 
        less than or equal to the column number.

    polyorder : int, optional
        The order of the polynomial used to fit the samples. Must be less than 
        ``window_length``. The default is `3`.

    deriv : int, optional
        The order of the derivative to compute. Must be a nonnegative integer. The 
        default is `0`, which means to filter the data without differentiating.

    delta : float, optional
        The spacing of the samples to which the filter will be applied. This is only 
        used if ``deriv`` > 0. Default is `1.0`.

    Returns
    -------
    P_fit : `~numpy.ndarray`
        Fitted spatial profile.
    
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        The input frame with updated cosmic ray mask.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])

    _validateBool(use_mask, 'use_mask')

    if dispersion_axis == 'row':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', False, use_mask, transpose=False)
    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', False, use_mask, transpose=True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Assume that unit of the trace is [pixel]
    if isinstance(trace1d, Spectrum1D):
        try:
            trcenter = trace1d.meta['header']['TRCENTER']
        except:
            trcenter = np.median(
                _validate1DArray(trace1d.flux.value, 'trace1d', n_col, True)
            )
    else:
        trcenter = np.median(_validate1DArray(trace1d, 'trace1d', n_col, True))

    # The total profile width is the sum of the two elements in ``profile_width``. If 
    # they have opposite signs, the whole aperture will be on the same side of the 
    # trace.
    profile_width = _validateAperture(profile_width, 'profile_width')
    profile_width[0] *= -1; profile_width.sort()

    profile_edges = np.round(trcenter + profile_width).astype(int)

    n_frac = profile_edges[1] + 1 - profile_edges[0]

    idx_frac = np.arange(n_frac)

    slice_frac = slice(profile_edges[0], profile_edges[1] + 1)

    # DS -- data with sky background subtracted (Horne 1986)
    DS = data_arr.copy()

    # V -- variance (Horne 1986)
    V = uncertainty_arr**2

    # f = Σ(D-S) --- object flux by summing along the spatial dimension (Horne 1986)
    f = DS[slice_frac].sum(axis=0)

    # Observed profile
    P_obs = DS / f

    # (X, Y) grid
    X = np.tile(idx_col, (n_frac, 1))
    Y = np.tile(idx_frac, (n_col, 1)).T

    P_fit = np.zeros_like(P_obs)
    master_mask = mask_arr.copy()

    P_fit[slice_frac] = signal.savgol_filter(
        P_obs[slice_frac], window_length=window_length, polyorder=polyorder, 
        deriv=deriv, delta=delta, axis=-1, mode='interp', cval=0.0)

    residual = P_obs[slice_frac] - P_fit[slice_frac]
    
    stddev = np.std(residual, axis=-1, ddof=1)
    threshold_lower, threshold_upper = -3 * stddev, 3 * stddev
    
    master_mask[slice_frac] = (
        (residual < threshold_lower[:, np.newaxis]) | 
        (threshold_upper[:, np.newaxis] < residual)
    )

    # Non-negative
    P_fit = np.max(np.stack([P_fit, np.zeros_like(P_fit)]), axis=0)
    # Normalization
    P_fit /= P_fit.sum(axis=0)

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'profile':
            title = f'{title} profile'

        fig, ax = plt.subplots(1, 1, dpi=100)

        _plot2d(ax=ax, ccd=P_fit, cmap='Greys_r', contrast=0.25, aspect='auto')
        (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        if dispersion_axis == 'row':
            for profile_edge in profile_edges:
                ax.axhline(y=profile_edge, ls='--', color='r', lw=1.5)
        else:
            for profile_edge in profile_edges:
                ax.axvline(x=profile_edge, ls='--', color='r', lw=1.5)

        # Settings
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=16)
        fig.set_figheight(n_row / n_col * fig.get_figwidth())
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    if save:

        fig_path = _validatePath(path, f'{title} fitting', '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in idx_frac:

                fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                fig.subplots_adjust(hspace=0)

                _plotFitting(
                    ax=ax, x=idx_col, y=P_obs[slice_frac][i], 
                    m=master_mask[slice_frac][i], x_fit=idx_col, 
                    y_fit=P_fit[slice_frac][i], r=residual[i], 
                    threshold_lower=threshold_lower[i], 
                    threshold_upper=threshold_upper[i], xlabel='dispersion axis [px]', 
                    ylabel='fraction', use_relative=False)

                # Settings
                ax[0]
                ax[0].set_title(f'profile at row {idx_row[slice_frac][i]}', fontsize=16)

                fig.align_ylabels()
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()

    nccd.mask = master_mask.copy()

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['PROFILE'] = '{} Uncertainty and mask refined.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return P_fit, nccd


def extract(ccd, dispersion_axis, method, trace1d=None, aper_width=None, n_aper=None, 
            profile2d=None, background2d=None, rdnoise=None, maxiters=None, 
            sigma_lower=None, sigma_upper=None, grow=None, spectral_axis=None, 
            use_uncertainty=True, use_mask=True, title='aperture', show=conf.fig_show, 
            save=conf.fig_save, path=conf.fig_path):
    """Extract 1-dimensional spectra.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    dispersion_axis : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.
    
    method : str
        Extraction method. `sum` or `optimal`, and
        - if `sum`, the source flux from the pixels within an aperture defined by 
        ``trace1d`` and ``aper_width`` are summed. The ``n_aper`` argument allows the 
        users to divide the whole aperture into several sub-apertures.
        - if `optimal`, the optimal extraction algorithm will be used. The normalized 
        spatial profile is used to compute the source flux. The sky background and 
        readout noise are used to estimate variance iteratively. ``n_aper`` is ignored.

    trace1d : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray` or `None`, optional
        Input trace (used by summation extraction algorithm).
    
    aper_width : scalar or 2-tuple or `None`, optional
        Aperture width (used by summation extraction algorithm).

    n_aper : int or `None`, optional
        Number of sub-apertures (used by the summation extraction algorithm).
        Default is `1`.

    profile2d : `~numpy.ndarray` or `None`, optional
        Spatial profile (used by the optimal extraction algorithm).

    background2d : `~numpy.ndarray`
        Pre-modeled sky background of ``ccd``. Used to estimate variance iteratively. 
        Note that this sky background should have already been subtraced from ``ccd``.

    rdnoise : scalar
        Readout noise of ``ccd``. Used to estimate variance iteratively.

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
    
    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits.

    spectral_axis : `~astropy.units.Quantity` or `None`
        Spectral axis of the extracted 1-dimensional spectra.

    use_uncertainty : bool, optional
        If `True` and an uncertainty array is attributed to ``ccd``, the uncertainties 
        are used as weights in the fitting. 
        Default is `False`.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    spectrum1d : `~specutils.Spectrum1D`
        Extracted 1-dimensional spectra.
    """

    _validateString(dispersion_axis, 'dispersion_axis', ['col', 'row'])
    
    if dispersion_axis == 'row':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', use_uncertainty, use_mask, False)
    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', use_uncertainty, use_mask, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    _validateString(method, 'method', ['sum', 'optimal'])

    variance_arr = uncertainty_arr**2

    if method == 'sum':

        # Assume that unit of the trace is [pixel]
        if isinstance(trace1d, Spectrum1D):
            trace1d = trace1d.flux.value
        trace1d = _validate1DArray(trace1d, 'trace1d', n_col, True)

        # The total aperture width is the sum of the two elements of ``aper_width``. If 
        # they have opposite signs, the whole aperture will be on the same side of the 
        # trace.
        aper_width = _validateAperture(aper_width, 'aper_width')
        aper_width[0] *= -1; aper_width.sort()
        
        _validateInteger(n_aper, 'n_aper', (1, None), (True, None))
        
        aper_edges = trace1d + np.linspace(
            aper_width[0], aper_width[1], n_aper + 1)[:, np.newaxis]

        # Out-of-range problem can be ignored in background modeling while cannot here.
        if (aper_edges.min() < -0.5) | (aper_edges.max() > n_row - 0.5):
            raise ValueError('Aperture edge is out of range.')

        data_aper = np.zeros((n_aper, n_col))
        variance_aper = np.zeros((n_aper, n_col))
        mask_aper = np.zeros((n_aper, n_col), dtype=bool)

        for i in idx_col:

            aper_start, aper_end = aper_edges[:-1, i], aper_edges[1:, i]

            for j in range(n_aper):

                # Internal pixels
                mask = (aper_start[j] < idx_row - 0.5) & (idx_row + 0.5 < aper_end[j])
                data_aper[j, i] = data_arr[mask, i].sum()
                variance_aper[j, i] = np.sum(variance_arr[mask, i])
                mask_aper[j, i] = np.any(mask_arr[mask, i])

                # Edge pixels

                # ``idx_start`` labels the pixel where ``aper_start`` is in
                idx_start = idx_row[idx_row - 0.5 <= aper_start[j]][-1]

                # ``idx_end`` labels the pixel where ``aper_end`` is in
                idx_end = idx_row[idx_row + 0.5 >= aper_end[j]][0]

                # ``aper_start`` and ``aper_end`` are in the same pixel
                if idx_start == idx_end:
                    data_aper[j, i] += (
                        data_arr[idx_end, i] * (aper_end[j] - aper_start[j]))
                    variance_aper[j, i] += (
                        variance_arr[idx_end, i] * (aper_end[j] - aper_start[j])**2)
                    mask_aper[j, i] |= mask_arr[idx_end, i]

                # in different pixels
                else:
                    data_aper[j, i] += (
                        data_arr[idx_start, i] * (idx_start + 0.5 - aper_start[j])
                        + data_arr[idx_end, i] * (aper_end[j] - (idx_end - 0.5))
                    )
                    variance_aper[j, i] += (
                        (variance_arr[idx_start, i]
                         * (idx_start + 0.5 - aper_start[j])**2)
                        + (variance_arr[idx_end, i] 
                           * (aper_end[j] - (idx_end - 0.5))**2)
                    )
                    mask_aper[j, i] |= mask_arr[idx_start, i] | mask_arr[idx_end, i]

        uncertainty_aper = np.sqrt(variance_aper)
    
    else:

        # P -- spatial profile (Horne 1986)
        P = _validateNDArray(profile2d, 'profile2d', data_arr.ndim)
        # S -- sky background (Horne 1986)
        S = _validateNDArray(background2d, 'background2d', data_arr.ndim)
        if not (data_arr.shape == P.shape == S.shape):
            raise ValueError(
                'The input frame, the spatial profile, and the sky background should '
                'have the same shape.'
            )

        _validateRange(rdnoise, 'rdnoise', (0, None), (True, None))

        _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

        # V -- variance (Horne 1986)
        V_inverse = (1 / variance_arr) if use_uncertainty else (1 + variance_arr)
        # In case of zero variance
        V_inverse[~np.isfinite(V_inverse)] = 0

        # M -- cosmic ray mask (Horne 1986)
        M_new = (~mask_arr) & (P > 0)
        M_old = np.zeros_like(M_new, dtype=bool)

        k = 0
        while np.any(M_new != M_old) & (k <= maxiters):

            M_old = M_new.copy()

            # f = (Σ M x (D-S)/P x P^2/V) / (Σ M x P^2/V) --- optimal object flux 
            # (Horne 1986)
            f = np.sum(M_old * data_arr * P * V_inverse, axis=0) / \
                np.sum(M_old * P**2 * V_inverse, axis=0)

            # var[f] = (Σ M x P) / (Σ M x P^2/V) --- variance of optimal 
            # object flux (Horne 1986)
            var_f = np.sum(M_old * P, axis=0) / np.sum(M_old * P**2 * V_inverse, axis=0)
            
            # Update
            V_inverse = 1 / (rdnoise**2 + np.abs(f * P + S))

            # Standardized residual
            residual = (data_arr - f * P) * np.sqrt(V_inverse)

            if maxiters == 0:
                threshold_lower, threshold_upper = None, None

            elif k < maxiters:
                residual[(mask_arr | (P == 0))] = np.nan
                residual_masked, threshold_lower, threshold_upper = sigma_clip(
                    data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                    maxiters=1, stdfunc='mad_std', axis=None, masked=True, 
                    return_bounds=True, grow=grow)
                M_new = ~residual_masked.mask

            k += 1

        data_aper = f[np.newaxis]
        uncertainty_aper = np.sqrt(var_f)[np.newaxis]
        mask_aper = np.zeros_like(data_aper, dtype=bool)
        
        aper_edges = np.zeros((2, n_col))
        for i in idx_col:
            aper_edges[:, i] = np.where(P[:, i] > 0)[0][[0, -1]]

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'aperture':
            title = f'{title} aperture'

        # Background image
        height_ratios = (1 / 4.5, n_col / n_row)
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=height_ratios, dpi=100)

        # Subplot 1
        ax[0].step(idx_row, np.nanmedian(data_arr, axis=1), 'k-', lw=1.5, where='mid')
        for aper_edge in aper_edges:
            ax[0].axvline(x=aper_edge.mean(), ls='--', color='r', lw=1.5)

        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_ylabel('pixel value', fontsize=16)
        ax[0].set_title(title, fontsize=16)

        # Subplot 2
        if dispersion_axis == 'row':
            xlabel, ylabel = 'row', 'column'
        else:
            xlabel, ylabel = 'column', 'row'
        _plot2d(
            ax=ax[1], ccd=data_arr.T, cmap='Greys_r', contrast=0.25, cbar=False, 
            xlabel=xlabel, ylabel=ylabel, aspect='auto')
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        for aper_edge in aper_edges:
            ax[1].plot(aper_edge, idx_col, 'r--', lw=1.5)

        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.set_figheight(fig.get_figwidth() * np.sum(height_ratios))
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    # Output
    spectrum1d = Spectrum1D(
        spectral_axis=spectral_axis, flux=(data_aper * nccd.unit), 
        uncertainty=StdDevUncertainty(uncertainty_aper), mask=mask_aper, 
        meta={'header': nccd.header})

    if n_aper == 1:
        spectrum1d = spectrum1d[0]

    return spectrum1d
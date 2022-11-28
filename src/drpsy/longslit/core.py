from copy import deepcopy
import os
import warnings

# NumPy
import numpy as np
# SciPy
from scipy import interpolate, signal, ndimage, spatial
from scipy.optimize import OptimizeWarning
# matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.table import Table
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from astropy.nddata import CCDData, StdDevUncertainty
# ccdproc
from ccdproc import flat_correct
# specutils
from specutils import Spectrum1D

from ..core import CCDDataList
from ..modeling import Poly1D, Spline1D, Spline2D, GaussianSmoothing2D
from ..plotting import plotFitting, _plot2d, _plotSpectrum1D
from ..validate import (_validateBool, _validateString, _validateRange, 
                        _validateInteger, _validate1DArray, _validateCCDList, 
                        _validateCCD, _validateSpectrum, _validateBins, 
                        _validateAperture, _validatePath)
from .utils import (_center1D_Gaussian, _refinePeakBases, _refinePeaks, loadSpectrum1D, 
                    loadStandardSpectrum, loadExtinctionCurve)

# Set plot parameters
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['response', 'illumination', 'align', 'fitcoords', 'dispcor', 'trace', 
           'background', 'extract', 'sensfunc']


def response(ccd, slit_along, n_piece=3, n_iter=5, sigma_lower=None, sigma_upper=None, 
             grow=False, use_mask=False, title='response', show=True, save=False, 
             path=None):
    """Determine response calibration.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned image, which is 
        always of the same shape as ``ccd``.

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
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Response calibration image, of the same type as the input ``ccd``.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    _validateBool(use_mask, 'use_mask')
    
    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    n_row, n_col = data_arr.shape

    # Apply mask
    data_arr[mask_arr] = np.nan

    # Average along spatial (slit) axis
    x = np.arange(n_col)
    # Bad pixels (NaNs or infs) in the original image (if any) may lead to unmasked 
    # elements in ``y`` and may cause an error in the spline fitting below.
    y = np.nanmean(data_arr, axis=0)
    mask_y = np.all(mask_arr, axis=0)

    # Fit cubic spline function
    spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
        x=x, y=y, weight=None, mask=mask_y, order=3, n_piece=n_piece, n_iter=n_iter, 
        sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, use_relative=True)
    y_fit = spl(x)
    uncertainty_y_fit = (y_fit * residual)[~master_mask].std(ddof=1)

    # Plot
    plotFitting(
        x=x, y=y, residual=residual, mask=master_mask, x_fit=x, y_fit=y_fit, 
        threshold_lower=threshold_lower, threshold_upper=threshold_upper, 
        xlabel='dispersion axis [px]', ylabel='pixel value', title='response', 
        show=show, save=save, path=path, use_relative=True)

    # Generate response calibrated image
    if slit_along == 'col':
        rccd = CCDData(
            data=np.tile(y_fit, (n_row, 1)) * nccd.unit, 
            uncertainty=StdDevUncertainty(np.full((n_row, n_col), uncertainty_y_fit)), 
            mask=np.tile(master_mask, (n_row, 1))
        )

    else:
        rccd = CCDData(
            data=np.tile(y_fit, (n_row, 1)).T * nccd.unit, 
            uncertainty=StdDevUncertainty(np.full((n_col, n_row), uncertainty_y_fit)), 
            mask=np.tile(master_mask, (n_row, 1).T)
        )

    nccd = flat_correct(ccd=nccd, flat=rccd, norm_value=1)

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['RESPONSE'] = '{} Response corrected.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd

# todo: improve multiplot, plot arrays.
def illumination(ccd, slit_along, method, sigma=None, n_piece=None, bins=5, n_iter=5, 
                 sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
                 title='illumination', show=True, save=False, path=None):
    """Model illumination.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned image, which is 
        always of the same shape as ``ccd``.

    method : str
        `Gaussian2D`, `CubicSpline2D`, or `iraf`.
        - if `Gaussian2D`, ``sigma`` is required. [...]
        - if `CubicSpline2D`, ``n_piece`` is required. [...]
        - if `iraf`, ``n_piece`` is required. [...]

    sigma : scalar or sequence of scalars
        Standard deviation for `~scipy.ndimage.gaussian_filter`.
        Default is `None`.

    n_piece : int or sequence of scalars
        Number of spline pieces. Lengths are all equal. Must be positive.
        - if methed is `CubicSpline2D`, [...]
        - if method is `iraf`, [...]
    
    bins : int or sequence of scalars, optional
        If `int`, it defines the number of equal-width bins. If a sequence, it defines 
        a monotonically increasing array of bin edges, including the rightmost edge, 
        allowing for non-uniform bin widths.
        
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
        the clipping limits.
    
    use_mask : bool, optional
        If `True`, and
        - if method is `Gaussian2D`, the masked pixels are interpolated before 
        convolution.
        - if method is `CubicSpline2D`, the masked pixels are ignored in the fitting.
        
    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Modeled illumination.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    _validateBool(use_mask, 'use_mask')
    
    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    n_row, n_col = data_arr.shape
    
    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # The largest allowed value for bin edge is set to ``n_col`` (or ``n_row`` 
    # depending on ``slit_along``) in order to enable the access to the last column (or 
    # row), i.e., data_arr[:, bin_edges[0][0]:bin_edges[0][1]].
    if slit_along == 'col':
        bin_edges = _validateBins(bins, n_col)
    else:
        bin_edges = _validateBins(bins, n_row)
    n_bin = bin_edges.shape[0]
    loc_bin = (bin_edges.sum(axis=1) - 1) / 2
    
    _validateString(method, 'method', ['Gaussian2D', 'CubicSpline2D', 'iraf'])

    if '2D' in method: # 2-dimensional method (`Gaussian2D` or `CubicSpline2D`)

        x = np.tile(idx_col, (n_row, 1))
        y = np.tile(idx_row, (n_col, 1)).T

        if method == 'Gaussian2D':

            if slit_along == 'col':

                try:
                    sigma = sigma[::-1]

                except:
                    pass

            data_fit, residual, threshold_lower, threshold_upper, master_mask = (
                GaussianSmoothing2D(
                    x=x, y=y, z=data_arr, sigma=sigma, mask=mask_arr, n_iter=n_iter, 
                    sigma_lower=sigma_lower, sigma_upper=sigma_upper, axis=0, 
                    grow=grow, use_relative=True)
            )

        elif method == 'CubicSpline2D':
            
            if slit_along == 'col':
                try:
                    n_piece = n_piece[::-1]

                except:
                    pass

            bispl, residual, threshold_lower, threshold_upper, master_mask = Spline2D(
                x=x, y=y, z=data_arr, weight=None, mask=mask_arr, order=(3, 3), 
                n_piece=n_piece, n_iter=n_iter, sigma_lower=sigma_lower, 
                sigma_upper=sigma_upper, axis=0, grow=grow, use_relative=True)

            data_fit = bispl(idx_col, idx_row, grid=True).T
        
        uncertainty_arr = (data_fit * residual)[~master_mask].std(ddof=1)

        # In 2D case, ``loc_bin`` is only used as index, thus converted to ``int``.
        idx_bin = loc_bin.astype(int)
        bin_data_arr = data_arr[:, idx_bin].T
        bin_data_fit = data_fit[:, idx_bin].T
        bin_mask_arr = master_mask[:, idx_bin].T
        bin_residual = residual[:, idx_bin].T
        if n_iter == 0:
            # Both are `None` originally
            bin_threshold_lower = [None] * n_bin
            bin_threshold_upper = [None] * n_bin
        else:
            # Both ``threshold_lower`` and ``threshold_upper`` are 2-dimensional 
            # arrays, though each array only has one row. Use flatten to get rid of 
            # the additional dimension.
            bin_threshold_lower = threshold_lower.flatten()[idx_bin]
            bin_threshold_upper = threshold_upper.flatten()[idx_bin]

    elif method == 'iraf':  # 1-dimensional method (`iraf`)

        n_piece = _validate1DArray(n_piece, 'n_piece', n_bin, True)

        # Apply mask
        data_arr[mask_arr] = np.nan

        uncertainty_arr = np.zeros_like(data_arr)
        master_mask = np.zeros_like(data_arr, dtype=bool)

        bin_data_arr = np.zeros((n_bin, n_row))
        bin_data_fit = np.zeros((n_bin, n_row))
        bin_mask_arr = np.zeros((n_bin, n_row), dtype=bool)
        bin_residual = np.zeros((n_bin, n_row))
        bin_threshold_lower = [None] * n_bin
        bin_threshold_upper = [None] * n_bin
        for i, (bin_start, bin_end) in enumerate(bin_edges):
            # Bad pixels (NaNs or infs) in the original image (if any) may lead to 
            # unmasked elements in ``count_bin_arr`` and may cause an error in the 
            # spline fitting below.
            bin_data_arr[i] = np.nanmean(data_arr[:, bin_start:bin_end], axis=1)
            bin_mask = np.all(mask_arr[:, bin_start:bin_end], axis=1)

            # Fit cubic spline function
            bin_spl, bin_residual[i], bin_threshold_lower[i], bin_threshold_upper[i], \
            bin_mask_arr[i] = Spline1D(
                x=idx_row, y=bin_data_arr[i], weight=None, mask=bin_mask, order=3, 
                n_piece=n_piece[i], n_iter=n_iter, sigma_lower=sigma_lower, 
                sigma_upper=sigma_upper, grow=grow, use_relative=True)
            bin_data_fit[i] = bin_spl(idx_row)
            uncertainty_arr[:, bin_start:bin_end] = (
                bin_data_fit[i] * bin_residual[i])[~bin_mask_arr[i]].std(ddof=1)
            master_mask[bin_mask_arr[i], bin_start:bin_end] = True
        
        # Interpolate
        data_fit = interpolate.interp1d(
            x=loc_bin, y=bin_data_fit.T, axis=1, kind='linear', bounds_error=False, 
            fill_value='extrapolate', assume_sorted=True)(idx_col)

    if slit_along != 'col':
        data_fit = data_fit.T
        uncertainty_arr = uncertainty_arr.T
        master_mask = master_mask.T
        n_col, n_row = n_row, n_col
        idx_col, idx_row = idx_row, idx_col

    # Plot
    if slit_along == 'col':
        x = idx_row
    else:
        x = idx_col

    for i in range(n_bin):
        plotFitting(
            x=x, y=bin_data_arr[i], residual=bin_residual[i], mask=bin_mask_arr[i], 
            x_fit=x, y_fit=bin_data_fit[i], threshold_lower=bin_threshold_lower[i], 
            threshold_upper=bin_threshold_upper[i], xlabel='spatial axis [px]', 
            ylabel='pixel value', title=f'illumination at {slit_along} {loc_bin[i]}', 
            show=show, save=save, path=path, use_relative=True)

    nccd.data = deepcopy(data_fit)

    if nccd.uncertainty is not None:
        nccd.uncertainty.array = deepcopy(uncertainty_arr)

    if nccd.mask is not None:
        nccd.mask[master_mask] = True

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKILLUM'] = '{} Illumination.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd


def align(ccdlist, slit_along, index=0):
    """Align multiple frames.
    
    Cross-correlation across dispersion axis is performed. Resolution is 1 [px].

    Parameters
    ----------
    ccdlist : Iterable
        Iterable object containing at least two 2-dimensional frames.

    slit_along : str
        `col` or `row`.
    
    index : int, optional
        Index of the reference frame.

    Returns
    -------
    out : `~drpsy.ccddatalist.CCDDataList`
        Aligned frames.
    """

    ccdlist = _validateCCDList(ccdlist, 2, 2)

    # Reference frame
    ccd_ref = ccdlist[index].copy()

    _validateString(slit_along, '``slit_along``', ['col', 'row'])

    if slit_along == 'col':
        axis = 1
    else:
        axis = 0

    # Median is used (instead of mean) to get rid of cosmic rays. This allows customers
    # to remove cosmic rays (after alignment) through frame combination. ``a`` does not 
    # contain any NaNs or infs in most cases, unless there are bad columns along 
    # dispersion axis.
    a = np.nanmedian(ccd_ref.data, axis=axis)

    x = np.arange(a.shape[0])
    
    # NaNs and infs are interpolated
    isFinite = np.isfinite(a)
    a = interpolate.interp1d(
        x=x[isFinite], y=a[isFinite], kind='linear', bounds_error=None, 
        fill_value=(a[isFinite][0], a[isFinite][-1]), assume_sorted=True)(x)

    a = np.hstack([a, a[:-1]])

    nccdlist = list()
    for ccd in ccdlist:

        # Frame to be aligned
        nccd = ccd.copy()

        # Median is used (instead of mean) to get rid of cosmic rays. This allows 
        # customers to remove cosmic rays (after alignment) through frame combination. 
        # ``v`` does not contain any NaNs or infs in most cases, unless there are bad 
        # columns along dispersion axis.
        v = np.nanmedian(nccd.data, axis=axis)

        # NaNs and infs are interpolated
        isFinite = np.isfinite(v)
        v = interpolate.interp1d(
            x=x[isFinite], y=v[isFinite], kind='linear', bounds_error=None, 
            fill_value=(v[isFinite][0], v[isFinite][-1]), assume_sorted=True)(x)

        # Cross-correlation
        shift = signal.correlate(a, v, mode='valid', method='auto').argmax()
        
        # Roll to align with the reference frame
        nccd.data = np.roll(nccd.data, shift, axis=(1 - axis))

        if nccd.uncertainty is not None:
            nccd.uncertainty.array = np.roll(
                nccd.uncertainty.array, shift, axis=(1 - axis))

        if nccd.mask is not None:
            nccd.mask = np.roll(nccd.mask, shift, axis=(1 - axis))

        nccd.header['ALIGNMEN'] = '{} Aligned (shift = {}).'.format(
            Time.now().to_value('iso', subfmt='date_hm'), shift)
        
        nccdlist.append(nccd)

    return CCDDataList(nccdlist)

# todo: improve multiplot
def fitcoords(ccd, slit_along, order=0, n_med=5, prominence=1e-3, n_piece=3, n_iter=5, 
              sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
              show=True, save=False, path=None, **kwargs):
    """Fit distortion across the slit.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned image, which is 
        always of the same shape as ``ccd``.
    
    order : int, optional
        Order of correction. Must be `2` >= ``order`` >= `0`.

    n_med : int, optional
        Number of pixels to median. Should be small for heavy distortion. Must be 
        positive.
        Default is `5`.
    
    prominence ：scalar or ndarray or sequence, optional
        Required prominence of peaks. see `~scipy.signal.find_peak` for details.

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
        the clipping limits.
    
    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        interpolated before cross-correlation. 
        Default is `False`.

    Returns
    -------
    U, V : `~numpy.ndarray`
        The coordinate maps U(X, Y) and V(X, Y).
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    _validateBool(use_mask, 'use_mask')
    
    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    n_row, n_col = data_arr.shape

    _validateInteger(n_med, 'n_med', (1, None), (True, None))
    
    # Split into bins along spatial (slit) axis
    bin_edges = np.hstack([np.arange(0, n_row, n_med), n_row])
    bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T
    n_bin = bin_edges.shape[0]
    loc_bin = (bin_edges.sum(axis=1) - 1) / 2

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)
    bin_data_arr = np.zeros((n_bin, n_col))

    # A precision of 0.05 px for cross-correlation (can be changed)
    n_sub = 20
    # Of type `~numpy.float64`, although still named as an index array.
    idx_col_dense = np.linspace(0, idx_col[-1], idx_col[-1] * n_sub + 1)
    bin_data_arr_dense = np.zeros((n_bin, idx_col_dense.shape[0]))

    # Apply mask
    data_arr[mask_arr] = np.nan

    for i, (bin_start, bin_end) in enumerate(bin_edges):

        bin_data = np.nanmean(data_arr[bin_start:bin_end, :], axis=0)
        bin_mask = np.all(mask_arr[bin_start:bin_end, :], axis=0)

        # Interpolate masked pixels
        fill_value = (bin_data[~bin_mask][0], bin_data[~bin_mask][-1])
        bin_data = interpolate.interp1d(
            x=idx_col[~bin_mask], y=bin_data[~bin_mask], bounds_error=False, 
            fill_value=fill_value, assume_sorted=True)(idx_col)

        # Normalize
        bin_data_arr[i] = bin_data / bin_data.max()
        
        # Interpolate to a denser grid
        bin_data_arr_dense[i] = interpolate.interp1d(
            x=idx_col, y=bin_data_arr[i], assume_sorted=True)(idx_col_dense)

    # Reference spectrum
    idx_ref = n_bin // 2
    bin_data_ref_dense = bin_data_arr_dense[idx_ref]

    # Extend reference spectrum
    # Here assumes that the zeropoint shift is no more than ±(n_col/4)
    n_ext = n_col // 4
    idx_ref_dense_ext = np.linspace(
        0, idx_col[-1] + 2 * n_ext, (idx_col[-1] + 2 * n_ext) * n_sub + 1) - n_ext
    bin_data_ref_dense_ext = np.hstack([
        bin_data_ref_dense[-n_ext * n_sub:], 
        bin_data_ref_dense, 
        bin_data_ref_dense[:n_ext * n_sub]
    ])

    shift_arr = np.zeros(n_bin)
    for i in range(n_bin):
        # Get zeropoint shift through cross-correlation
        idx_max = signal.correlate(
            bin_data_ref_dense_ext, bin_data_arr_dense[i], mode='valid', method='auto'
        ).argmax()
        shift_arr[i] = idx_ref_dense_ext[idx_max]

    _validateInteger(order, 'order', (0, 2), (True, True))

    if order == 0:

        # Univariate cubic spline fitting
        spl, residual, threshold_lower, threshold_upper, mask = Spline1D(
            x=loc_bin, y=shift_arr, weight=None, mask=None, order=3, n_piece=n_piece, 
            n_iter=n_iter, sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)
        shift_fit = spl(idx_row)

        U = idx_col + shift_fit[:, np.newaxis]

        plotFitting(
            x=loc_bin, y=shift_arr, residual=residual, mask=mask, x_fit=idx_row, 
            y_fit=shift_fit, threshold_lower=threshold_lower, 
            threshold_upper=threshold_upper, xlabel='spatial axis [px]', 
            ylabel='shift value [px]', title='zeropoint shift curve', show=show, 
            save=save, path=path, use_relative=False)

    else:

        shift = np.round(shift_arr * n_sub).astype(int)
        rolled_bin_data_arr_dense = np.zeros_like(bin_data_arr_dense)
        for i in range(n_bin):
            # Roll to align with reference spectrum
            rolled_bin_data_arr_dense[i] = np.roll(bin_data_arr_dense[i], shift[i])
        bin_data_mean = rolled_bin_data_arr_dense.mean(axis=0)[::n_sub]

        # Peak detection
        shift_max, shift_min = shift_arr.max(), shift_arr.min()

        if shift_max <= 0:
            idx_start = 0
        else:
            idx_start = int(np.ceil(shift_max))

        if shift_min >= 0:
            idx_end = n_col
        else:
            idx_end = int(np.floor(shift_min))

        # Find peaks in the [idx_start, idx_end) range of reference bin to ensure that 
        # all the detected peaks are in the common range of each bin.
        peaks, properties = signal.find_peaks(
            x=bin_data_mean[idx_start:idx_end], prominence=prominence, **kwargs)

        n_peak = len(peaks)

        if n_peak == 0:
            raise RuntimeError('No peak detected.')

        # Too few peaks may lead to failure in the fitting below.
        elif n_peak < (order + 1) * (3 + 1):
            raise RuntimeError(
                'Too few peaks detected. Should be >= {} for order ({}, 3).'.format(
                    (order + 1) * (3 + 1), order))
            
        refined_left_bases, refined_right_bases = _refinePeakBases(
            peaks=peaks, left_bases=properties['left_bases'], 
            right_bases=properties['right_bases'], n_peak=n_peak, copy=True)

        heights = bin_data_mean[idx_start:idx_end][peaks]

        refined_peaks, refined_index = _refinePeaks(
            bin_data_mean[idx_start:idx_end], peaks, heights, refined_left_bases, 
            refined_right_bases, 1)

        peaks = refined_peaks + idx_start
        heights = heights[refined_index]
        left_bases = refined_left_bases[refined_index] + idx_start
        right_bases = refined_right_bases[refined_index] + idx_start

        refined_peaks_arr = np.full((n_bin, n_peak), np.nan)

        for i in range(n_bin):

            # Offset peak properties
            shifted_peaks = peaks - shift_arr[i]
            shifted_left_bases = left_bases - shift_arr[i]
            shifted_right_bases = right_bases - shift_arr[i]

            # Refine peaks
            refined_peaks, refined_index = _refinePeaks(
                bin_data_arr[i], shifted_peaks, heights, shifted_left_bases, 
                shifted_right_bases, 1)

            refined_peaks_arr[i, refined_index] = refined_peaks

        # Fitcoords
        x = deepcopy(refined_peaks_arr)
        y = np.tile(loc_bin, (n_peak, 1)).T
        z = np.tile(peaks, (n_bin, 1))
        mask = np.isnan(x) | np.isnan(z)
        # All the elements of ``x`` and ``y`` are within [idx_col[0], idx_col[-1]] and 
        # [idx_row[0], idx_row[-1]], respectively. Therefore, extrapolation is 
        # inevitably used when calculating z(x, y), which is defined on the whole 
        # image. However, `~scipy.interpolate.LSQBivariateSpline` called by `Spline2D` 
        # does not support extrapolation. An alternative way is to use ``bbox``. Here 
        # we set ``bbox`` to be the boundary of the whole image and therefore will get 
        # correct return when calling `~scipy.interpolate.LSQBivariateSpline.__call__`. 
        # Note that doing this will not change the fitting result as long as the 
        # internal knots remain the same. (Also see cautions mentioned by 
        # https://stackoverflow.com/questions/45904929.)
        bbox = [idx_col[0], idx_col[-1], idx_row[0], idx_row[-1]]
        bispl, residual, threshold_lower, threshold_upper, master_mask = Spline2D(
            x=x, y=y, z=z, weight=None, mask=mask, order=(order, 3), 
            n_piece=(1, n_piece), bbox=bbox, n_iter=n_iter, sigma_lower=sigma_lower, 
            sigma_upper=sigma_upper, axis=None, grow=grow, use_relative=False)

        # !!! Extrapolation is used here (see above) !!!
        U = bispl(idx_col, idx_row, grid=True).T

        _validateBool(show, 'show')

        title = ['peak detection', 'distortion residual']

        fig_path = _validatePath(save, path, title)

        if show | save:

            fig = plt.figure(figsize=(6, 6), dpi=100)
            # Split into subplots
            n_subplot = 2
            length = n_col // n_subplot + 1
            for i in range(n_subplot):
                idx_start, idx_end = i * length, (i + 1) * length
                idx_peak = np.where((idx_start <= peaks) & (peaks < idx_end))[0]
                ax = fig.add_subplot(n_subplot, 1, i + 1)
                ax.step(
                    idx_col[idx_start:idx_end], bin_data_mean[idx_start:idx_end], 
                    color='k', ls='-', where='mid')
                for idx in idx_peak:
                    ymin = heights[idx] * 1.2
                    ymax = heights[idx] * 1.5
                    ax.plot([peaks[idx], peaks[idx]], [ymin, ymax], 'r-', lw=1.5)
                # Settings
                ax.grid(axis='both', color='0.95', zorder=-1)
                ax.set_xlim(idx_start, idx_end)
                ax.set_yscale('log')
                ax.tick_params(
                    which='major', direction='in', top=True, right=True, length=5, 
                    width=1.5, labelsize=12)
                ax.tick_params(
                    which='minor', direction='in', top=True, right=True, length=3, 
                    width=1.5, labelsize=12)
                ax.set_ylabel('normalized intensity', fontsize=16)
            ax.set_xlabel('dispersion axis [px]', fontsize=16)
            fig.align_ylabels()
            fig.suptitle(title[0], fontsize=16)
            fig.tight_layout()

            if save: plt.savefig(fig_path[0], dpi=100)

            if show: plt.show()
            
            plt.close()

            if n_bin // 10 >= 10:
                idx_plot = np.linspace(0, n_bin - 1, 11).astype(int)

            else:
                idx_plot = np.arange(0, n_bin, 10)
                if idx_plot[-1] != n_bin - 1:
                    idx_plot = np.hstack([idx_plot, n_bin - 1])

            z_fit = bispl(idx_col, loc_bin, grid=True).T

            for i in idx_plot:
                plotFitting(
                    x=refined_peaks_arr[i], y=peaks - refined_peaks_arr[i], 
                    residual=residual[i], mask=master_mask[i], x_fit=idx_col, 
                    y_fit=z_fit[i] - idx_col, threshold_lower=threshold_lower, 
                    threshold_upper=threshold_upper, xlabel='dispersion axis [px]', 
                    ylabel='shift [px]', title=f'distortion at bin {loc_bin[i]}', 
                    show=show, save=save, path=path, use_relative=False)

            residual[master_mask] = np.nan
            cmap = plt.cm.get_cmap('Greys_r').copy(); cmap.set_bad('red', 1.)
            extent = (0.5, residual.shape[1] + 0.5, 0.5, residual.shape[0] + 0.5)

            fig = plt.figure(figsize=(6, 6), dpi=100)
            ax = fig.add_subplot(1, 1, 1)
            _plot2d(
                ax, residual, cmap=cmap, contrast=0.25, extent=extent, cbar=True, 
                xlabel='peak number', ylabel='bin number', cblabel='pixel')
            # Settings
            fig.suptitle(title[1], fontsize=16)
            fig.tight_layout()

            if save: plt.savefig(fig_path[1], dpi=100)

            if show: plt.show()

            plt.close()

    V = np.tile(idx_row, (n_col, 1)).T

    # Output
    if slit_along != 'col':
        U, V = V.T, U.T

    return U, V


def dispcor(spectrum1d, reverse, file_name, n_piece=3, refit=True, n_iter=5, 
            sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
            title='dispcor', show=True, save=False, path=None):
    """Dispersion correction.

    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    reverse : bool
        Reverse the input spectrum or not. Set to `True` if the input spectrum is in 
        the reverse order with respect to the reference spectrum.

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
    if np.ndim(flux) > 1:
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
    spectral_axis_unit = spectrum1d_ref.spectral_axis.unit.to_string('fits')
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
            xlabel=f'spectral axis [{spectral_axis_unit}]', 
            ylabel='dispersion axis [px]', title='dispersion solution', show=show, 
            save=save, path=path, use_relative=False)

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'dispcor':
        title = f'{title} dispcor'

    fig_path = _validatePath(save, path, title)

    if show | save:

        xlabel = 'spectral axis [{}]'.format(
            spectrum1d_ref.spectral_axis.unit.to_string('fits'))

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
        ax.set_xlabel(f'spectral axis [{spectral_axis_unit}]', fontsize=16)
        ax.set_ylabel('flux', fontsize=16)
        ax.legend(fontsize=16)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()
        
        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    if reverse:
        spectral_axis = spectral_axis[::-1]

    spectral_axis *= spectrum1d_ref.spectral_axis.unit

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


def trace(ccd, slit_along, fwhm, method, interval=None, n_med=3, n_piece=3, n_iter=5, 
          sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
          title='trace', show=True, save=False, path=None):
    """Trace on the 2-dimensional spectrum.
    
    First a 1-dimensional median array is generated by taking median along the 
    dispersion axis. Then the center of the specified feature (that is the strongest 
    one in the specified interval of the median array) is determined by fitting a 
    Gaussian profile. [...]
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.

    fwhm : scalar
        Estimated full width at half maximum of the peak to be traced.

    method : str
        `center` or `trace`, and
        - if `center`, a straight trace at the center of the specified feature is 
        returned.
        - if `trace`, the center of the specified feature is treated as an initial
        guess for bin by bin (defined according to ``n_med``) Gaussian fittings.
    
    interval : 2-tuple or `None`, optional
        Interval the specified feature lies in, and
        - if `None`, the brightest source in the frame is traced.
        - if 2-tuple, the brightest source in the interval is traced. (Use this when 
        the source is not the brightest one in the frame)
    
    n_med : int, optional
        Number of pixels to median. Large number for faint source.
        Default is `3`.
    
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
    
    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits.
    
    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    fitted_trace : 
        Fitted trace.
    
    center_med : scalar
        Center of the specified feature.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    if interval is None:
        interval = (0, None)
    else:
        interval = _validate1DArray(interval, 'interval', 2, False)

    _validateRange(fwhm, 'fwhm', (2, None), (True, None))

    # Bad pixels (NaNs or infs) in the original image (if any) may lead to unmasked 
    # elements in ``count_med`` and may cause an error in the Gaussian fitting below.
    count_med = np.nanmedian(data_arr, axis=1)
    mask_med = np.all(mask_arr, axis=1)

    # Peak center
    idx_src_med = interval[0] + np.ma.argmax(
        np.ma.array(count_med, mask=mask_med)[interval[0]:interval[1]], 
        fill_value=-np.inf)

    # Peak range
    idx_min_med = int(idx_src_med - 0.5 * fwhm)
    idx_max_med = int(idx_src_med + 0.5 * fwhm)

    # Initial guess
    initial_guess = (1, idx_src_med, fwhm)

    x_med = idx_min_med + np.arange(idx_max_med + 1 - idx_min_med)
    # Normalize
    y_med = count_med[idx_min_med:(idx_max_med + 1)] / count_med[idx_src_med]
    m_med = mask_med[idx_min_med:(idx_max_med + 1)]

    # Gaussian fitting
    center_med = _center1D_Gaussian(x_med[~m_med], y_med[~m_med], initial_guess, 0)

    _validateString(method, 'method', ['center', 'trace'])

    if method == 'center':

        fitted_trace = np.full(n_col, center_med)

    else:

        _validateInteger(n_med, 'n_med', (1, None), (True, None))

        # Split into bins along dispersion axis
        bin_edges = np.arange(0, n_col, n_med)
        if bin_edges[-1] < (n_col - 1):
            bin_edges = np.hstack([bin_edges, n_col - 1])
        bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T
        n_bin = bin_edges.shape[0]
        loc_bin = (bin_edges.sum(axis=1) - 1) / 2

        refined_trace = np.full(n_bin, np.nan)

        # Update fitting range
        idx_min_med = int(center_med - 2.5 * fwhm)
        idx_max_med = int(center_med + 2.5 * fwhm)

        for i in range(n_bin):

            # Bad pixels (NaNs or infs) in the original image (if any) may lead to 
            # unmasked elements in ``count_med`` and may cause an error in the 
            # Gaussian fitting below.
            count_bin = np.nanmedian(
                data_arr[:, bin_edges[i][0]:bin_edges[i][-1]], axis=1)
            mask_bin = np.all(mask_arr[:, bin_edges[i][0]:bin_edges[i][-1]], axis=1)

            # Peak center
            idx_src_bin = idx_min_med + np.ma.argmax(
                np.ma.array(count_bin, mask=mask_bin)[idx_min_med:idx_max_med], 
                fill_value=-np.inf)

            # Peak range
            idx_min_bin = int(idx_src_bin - 0.5 * fwhm)
            idx_max_bin = int(idx_src_bin + 0.5 * fwhm)

            # Initial guess
            initial_guess = (1, idx_src_bin, fwhm)

            x_bin = idx_min_bin + np.arange(idx_max_bin + 1 - idx_min_bin)
            # Normalize
            y_bin = count_bin[idx_min_bin:(idx_max_bin + 1)] / count_bin[idx_src_bin]
            m_bin = mask_bin[idx_min_bin:(idx_max_bin + 1)]

            # Gaussian fitting
            try:
                refined_trace[i] = _center1D_Gaussian(
                    x_bin[~m_bin], y_bin[~m_bin], initial_guess, 0)
            except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
                pass

        mask = np.isnan(refined_trace)

        # Spline fitting
        spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
            x=loc_bin, y=refined_trace, weight=None, mask=mask, order=3, n_piece=n_piece, 
            n_iter=n_iter, sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)

        fitted_trace = spl(idx_col)

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'trace':
        title = f'{title} trace'

    fig_path = _validatePath(save, path, title)

    if show | save:
        
        if method == 'trace':
            plotFitting(
                x=loc_bin, y=refined_trace, residual=residual, mask=master_mask, 
                x_fit=idx_col, y_fit=fitted_trace, threshold_lower=threshold_lower, 
                threshold_upper=threshold_upper, xlabel='dispersion axis [px]', 
                ylabel='spatial axis [px]', title=title, show=show, save=save, 
                path=path, use_relative=False)

        extent = (0.5, data_arr.shape[1] + 0.5, 0.5, data_arr.shape[0] + 0.5)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        gs = gridspec.GridSpec(3, 1)
        ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1:])
        # Subplot 1
        ax[0].step(idx_row, count_med, 'k-', lw=1.5, where='mid')
        ax[0].axvline(x=center_med, color='r', ls='--', lw=1.5)
        if method == 'trace':
            ax[0].axvline(x=idx_min_med, color='b', ls='--', lw=1.5)
            ax[0].axvline(x=idx_max_med, color='b', ls='--', lw=1.5)
        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
#         ax[0].set_yscale('log')
        ax[0].set_xlim(center_med - 10 * fwhm, center_med + 10 * fwhm)
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_xlabel('spatial axis [px]', fontsize=16)
        ax[0].set_ylabel('pixel value', fontsize=16)
        # Subplot 2
        _plot2d(
            ax=ax[1], ccd=data_arr, cmap='Greys_r', contrast=0.25, extent=extent, 
            cbar=False)
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        ax[1].plot(idx_col + 1, fitted_trace + 1, 'r-', lw=1.5)
        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    header = deepcopy(nccd.header)
    header['TRINTERV'] = f'{interval}'
    header['TRCENTER'] = center_med
    header['TRACE'] = '{} Trace ({}, n_med = {}, n_piece = {})'.format(
        Time.now().to_value('iso', subfmt='date_hm'), method, n_med, n_piece)
    meta = {'header': header}

    trace1d = Spectrum1D(flux=(fitted_trace * u.pixel), meta=meta)

    return trace1d


def background(ccd, slit_along, trace, distance=50, aper_width=50, degree=1, n_iter=5, 
               sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
               title='background', show=True, save=False, path=None):
    """Model background.
    
    Sky background of the input frame is modeled col by col (or row by row, depending 
    on the ``slit_along``) through polynomial fittings.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.
    
    trace : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray`
        Input trace.
    
    distance : scalar or 2-tuple, optional
        Distances from the input trace to the centers of the two background apertures.
    
    aper_width : scalar or 2-tuple, optional
        Aperture widths of the two background apertures.
        
    degree : int, optional
        Degree of the fitting polynomial.
        Default is `1`.
    
    n_iter : int, optional
        Number of sigma slipping iterations. Must be >= `0`. 
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

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Modeled background.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)
    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Assume that unit of the trace is [pixel]
    if isinstance(trace, Spectrum1D):
        trace = trace.flux.value
    trace = _validate1DArray(trace, 'trace', n_col, True)

    distance = _validate1DArray(distance, 'distance', 2, True)

    aper_width = _validate1DArray(aper_width, 'aper_width', 2, True)
    for i in range(2):
        if aper_width[i] < 1:
            raise ValueError('``aper_width`` should be >= `1`.')

    # Usually there are two separate background apertures (though they can be set 
    # overlapped). ``distance`` controls separations on either side of the trace.
    idx_lbkg_min = trace - distance[0] - aper_width[0] / 2
    idx_lbkg_max = trace - distance[0] + aper_width[0] / 2
    idx_rbkg_min = trace + distance[1] - aper_width[1] / 2
    idx_rbkg_max = trace + distance[1] + aper_width[1] / 2
    
    if (
        (idx_lbkg_min.min() < -0.5) | (idx_lbkg_max.max() > (n_row - 0.5)) | 
        (idx_rbkg_min.min() < -0.5) | (idx_rbkg_max.max() > (n_row - 0.5))
    ):
        raise warnings.warn('Background index out of range.', RuntimeWarning)
    
    bkg_arr = np.zeros_like(data_arr)
    std_arr = np.zeros_like(data_arr)

    # Background fitting
    for i in range(n_col):

        mask_bkg = (
            ((idx_lbkg_min[i] < idx_row) & (idx_row < idx_lbkg_max[i])) | 
            ((idx_rbkg_min[i] < idx_row) & (idx_row < idx_rbkg_max[i]))
        )

        p, residual, threshold_lower, threshold_upper, master_mask = Poly1D(
            x=idx_row[mask_bkg], y=data_arr[mask_bkg, i], weight=None, 
            mask=mask_arr[mask_bkg, i], degree=degree, n_iter=n_iter, 
            sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)

        bkg_arr[:, i] = p(idx_row)
        mask_arr[mask_bkg, i][master_mask] = True
        std_arr[:, i] = residual.std(ddof=1)

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'background':
        title = f'{title} background'

    fig_path = _validatePath(save, path, title)

    if show | save:

        extent = (0.5, nccd.shape[1] + 0.5, 0.5, nccd.shape[0] + 0.5)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        gs = gridspec.GridSpec(3, 1)
        ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1:])
        # Subplot 1
        ax[0].step(idx_row, np.nanmedian(data_arr, axis=1), 'k-', lw=1.5, where='mid')
        ax[0].axvline(x=idx_lbkg_min.mean(), color='y', ls='--', lw=1.5)
        ax[0].axvline(x=idx_lbkg_max.mean(), color='y', ls='--', lw=1.5)
        ax[0].axvline(x=idx_rbkg_min.mean(), color='b', ls='--', lw=1.5)
        ax[0].axvline(x=idx_rbkg_max.mean(), color='b', ls='--', lw=1.5)
        # Settings
        ax[0].grid(axis='both', color='0.95', zorder=-1)
        ax[0].set_xlim(idx_row[0], idx_row[-1])
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].set_xlabel('spatial axis [px]', fontsize=16)
        ax[0].set_ylabel('pixel value', fontsize=16)
        ax[0].set_title(title, fontsize=16)
        # Subplot 2
        _plot2d(
            ax=ax[1], ccd=nccd.data, cmap='Greys_r', contrast=0.25, extent=extent, 
            cbar=False)
        (xmin, xmax), (ymin, ymax) = ax[1].get_xlim(), ax[1].get_ylim()
        if slit_along == 'col':
            ax[1].plot(idx_col + 1, trace + 1, 'r--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_lbkg_min + 1, 'y--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_lbkg_max + 1, 'y--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_rbkg_min + 1, 'b--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_rbkg_max + 1, 'b--', lw=1.5)
        else:
            ax[1].plot(trace + 1, idx_row + 1, 'r--', lw=1.5)
            ax[1].plot(idx_lbkg_min + 1, idx_row + 1, 'y--', lw=1.5)
            ax[1].plot(idx_lbkg_max + 1, idx_row + 1, 'y--', lw=1.5)
            ax[1].plot(idx_rbkg_min + 1, idx_row + 1, 'b--', lw=1.5)
            ax[1].plot(idx_rbkg_max + 1, idx_row + 1, 'b--', lw=1.5)
        # Settings
        ax[1].set_xlim(xmin, xmax)
        ax[1].set_ylim(ymin, ymax)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    # Generate background image
    if slit_along == 'col':
        nccd.data = deepcopy(bkg_arr)
        if nccd.uncertainty is not None:
            nccd.uncertainty.array = deepcopy(std_arr)
        if nccd.mask is not None:
            nccd.mask = deepcopy(mask_arr)
    else:
        nccd.data = deepcopy(bkg_arr).T
        if nccd.uncertainty is not None:
            nccd.uncertainty.array = deepcopy(std_arr).T
        if nccd.mask is not None:
            nccd.mask = deepcopy(mask_arr).T

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKBACKGR'] = '{} Background.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd


def extract(ccd, slit_along, trace, aper_width, n_aper=1, spectral_axis=None, 
            title='aperture', show=True, save=False, path=None):
    """Extract 1-dimensional spectra.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.
    
    trace : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray`
        Input trace.
    
    aper_width : scalar or 2-tuple, optional
        Aperture width.
    
    n_aper : int, optional
        Number of sub-apertures.
        Default is `1`.

    spectral_axis : `~astropy.units.Quantity` or `None`
        Spectral axis of the extracted 1-dimensional spectra.

    Returns
    -------
    spectrum1d : `~specutils.Spectrum1D`
        Extracted 1-dimensional spectra.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    if slit_along == 'col':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', True, True, False)
    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', True, True, True)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Assume that unit of the trace is [pixel]
    if isinstance(trace, Spectrum1D):
        trace = trace.flux.value
    trace = _validate1DArray(trace, 'trace', n_col, True)

    aper_width = _validateAperture(aper_width)
    
    _validateInteger(n_aper, 'n_aper', (1, None), (True, None))

    # The total aperture width is the sum of ``aper_width[0]`` and ``aper_width[1]``. 
    # If they have opposite signs, the whole aperture will be on one side of the trace.
    aper_edges = trace + np.linspace(
        -aper_width[0], aper_width[1], n_aper + 1)[:, np.newaxis]

    # Out-of-range problem can be ignored in background modeling while cannot here.
    if (aper_edges.min() < -0.5) | (aper_edges.max() > n_row - 0.5):
        raise ValueError('Aperture edge is out of range.')

    data_aper = np.zeros((n_aper, n_col))

    uncertainty_aper = np.zeros((n_aper, n_col)) # !!! Variance !!!
    
    mask_aper = np.zeros((n_aper, n_col), dtype=bool)

    for i in idx_col:

        aper_start, aper_end = aper_edges[:-1, i], aper_edges[1:, i]

        for j in range(n_aper):

            # Internal pixels
            mask = (aper_start[j] < idx_row - 0.5) & (idx_row + 0.5 < aper_end[j])
            data_aper[j, i] = data_arr[mask, i].sum()
            uncertainty_aper[j, i] = np.sum(uncertainty_arr[mask, i]**2)
            mask_aper[j, i] = np.any(mask_arr[mask, i])

            # Edge pixels

            # ``idx_start`` labels the pixel where ``aper_start`` is in
            idx_start = idx_row[idx_row - 0.5 <= aper_start[j]][-1]

            # ``idx_end`` labels the pixel where ``aper_end`` is in
            idx_end = idx_row[idx_row + 0.5 >= aper_end[j]][0]
            
            # ``aper_start`` and ``aper_end`` are in the same pixel
            if idx_start == idx_end:
                data_aper[j, i] += data_arr[idx_end, i] * (aper_end[j] - aper_start[j])
                uncertainty_aper[j, i] += (
                    uncertainty_arr[idx_end, i] * (aper_end[j] - aper_start[j]))**2
                mask_aper[j, i] |= mask_arr[idx_end, i]

            # in different pixels
            else:
                data_aper[j, i] += (
                    data_arr[idx_start, i] * (idx_start + 0.5 - aper_start[j])
                    + data_arr[idx_end, i] * (aper_end[j] - (idx_end - 0.5))
                )
                uncertainty_aper[j, i] += (
                    (uncertainty_arr[idx_start, i]
                     * (idx_start + 0.5 - aper_start[j]))**2
                    + (uncertainty_arr[idx_end, i]
                       * (aper_end[j] - (idx_end - 0.5)))**2
                )
                mask_aper[j, i] |= mask_arr[idx_start, i] | mask_arr[idx_end, i]

    uncertainty_aper = np.sqrt(uncertainty_aper) # Standard deviation

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'aperture':
        title = f'{title} aperture'

    fig_path = _validatePath(save, path, title)

    if show | save:

        extent = (0.5, nccd.shape[1] + 0.5, 0.5, nccd.shape[0] + 0.5)

        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        _plot2d(ax=ax, ccd=nccd.data, cmap='Greys_r', contrast=0.25, extent=extent)
        (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        if slit_along == 'col':
            for aper_edge in aper_edges:
                ax.plot(idx_col + 1, aper_edge + 1, 'r--', lw=1.5)
        else:
            for aper_edge in aper_edges:
                ax.plot(aper_edge + 1, idx_row + 1, 'r--', lw=1.5)
        # Settings
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()

    # Output
    spectrum1d = Spectrum1D(
        spectral_axis=spectral_axis, flux=(data_aper * nccd.unit), 
        uncertainty=StdDevUncertainty(uncertainty_aper), mask=mask_aper, 
        meta={'header': nccd.header})

    return spectrum1d


# todo: slit loss.
def sensfunc(spectrum1d, exptime, airmass, extinct, standard, bandwid=None, 
             bandsep=None, n_piece=3, n_iter=5, sigma_lower=None, sigma_upper=None, 
             grow=False, use_mask=False, title='sensfunc', show=True, save=False, 
             path=None):
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

    bandwid, bandsep : scalar or `None`
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

    # Negative values lead to NaN here
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

    unit_sens_fit = new_spectrum1d.flux.unit / (u.erg / u.cm**2)

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


def _calibrate(sens1d, airmass, extinct, shape, use_uncertainty):
    """Apply a flux calibration"""

    new_sens1d, _, sens, uncertainty, _ = _validateSpectrum(
        sens1d, 'sens1d', use_uncertainty, False)

    # ``sens``, ``uncertainty`` and ``mask_sens`` should be 1-dimensional arrays. 
    # Use flatten to get rid of additional dimensions.
    if sens.ndim > 1:
        sens = sens.flatten()[:sens.shape[-1]]
        uncertainty = uncertainty.flatten()[:uncertainty.shape[-1]]

    wavelength = deepcopy(new_sens1d.wavelength)
    bin_width = np.abs(np.diff(new_sens1d.bin_edges.to(u.AA)))

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
            fill_value='extrapolate', assume_sorted=True)(wavelength.value)

        sens -= (airmass * extinction)

    sens = 10**(0.4 * sens)
    uncertainty *= 0.4 * np.log(10) * sens

    shape = np.hstack([shape, 1])

    sens = np.tile(sens, shape) * new_sens1d.flux.unit
    uncertainty = StdDevUncertainty(np.tile(uncertainty, shape))

    if 'header' in sens1d.meta:
        meta = deepcopy(sens1d.meta)
    else:
        meta = {'header': dict()}
    # Add headers here
    meta['header']['AIRMASS'] = airmass

    return wavelength, bin_width, sens, uncertainty, meta

# todo: uncertainty
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

    if new_spectrum1d.flux.ndim == 1:
        shape = []
    else:
        shape = new_spectrum1d.flux.shape[:-1]

    _validateBool(use_uncertainty, 'use_uncertainty')

    wavelength, bin_width, sens, uncertainty_sens, meta_sens = _calibrate(
        sens1d, airmass, extinct, shape, use_uncertainty)

    if isinstance(exptime, str):
        exptime = spectrum1d.meta['header'][exptime] * u.s

    flux_obs = new_spectrum1d.flux / (exptime * bin_width) # [counts/s/Angstrom]

    if new_spectrum1d.uncertainty is not None:
        uncertainty_obs = StdDevUncertainty(
            new_spectrum1d.uncertainty.array / (exptime.value * bin_width.value))
    else:
        uncertainty_obs = None

    new_sens1d = Spectrum1D(
        spectral_axis=wavelength, flux=sens, uncertainty=uncertainty_sens, 
        meta=meta_sens)

    new_spectrum1d = Spectrum1D(
        spectral_axis=wavelength, flux=flux_obs, uncertainty=uncertainty_obs, 
        mask=new_spectrum1d.mask, meta=new_spectrum1d.meta)

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
    meta['header']['AIRMASS'] = new_sens1d.meta['header']['AIRMASS']
    meta['header']['CALIBRAT'] = '{} Calibrated'.format(
        Time.now().to_value('iso', subfmt='date_hm'))

    calibrated_spectrum1d.meta = meta

    return calibrated_spectrum1d


def calibrate2d(ccd, slit_along, exptime, airmass, extinct, sens1d, 
                use_uncertainty=False):
    """Apply a flux calibration to a 2-dimensional spectrum.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned image, which is 
        always of the same shape as ``ccd``.

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
    calibrated_ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Calibrated 2-dimensional spectrum.
    """
    
    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    if slit_along == 'col':
        nccd, data_arr, uncertainty_arr, _ = _validateCCD(
            ccd, 'ccd', True, False, False)
    else:
        nccd, data_arr, uncertainty_arr, _ = _validateCCD(
            ccd, 'ccd', True, False, True)

    if data_arr.ndim == 1:
        shape = []
    else:
        shape = data_arr.shape[:-1]

    _validateBool(use_uncertainty, 'use_uncertainty')

    wavelength, bin_width, sens, uncertainty_sens, meta_sens = _calibrate(
        sens1d, airmass, extinct, shape, use_uncertainty)

    # Unit conversion
    if isinstance(exptime, str):
        exptime = nccd.header[exptime] * u.s

    flux_obs = (data_arr * nccd.unit) / (exptime * bin_width) # [counts/s/Angstrom]
    uncertainty_obs = StdDevUncertainty(
        uncertainty_arr / (exptime.value * bin_width.value))

    if slit_along == 'col':

        sccd = CCDData(
            data=sens, uncertainty=uncertainty_sens, header=meta_sens['header'])

        nccd = CCDData(
            data=flux_obs, uncertainty=uncertainty_obs, mask=nccd.mask, 
            header=nccd.header)
    else:

        sccd = CCDData(
            data=sens.T, uncertainty=StdDevUncertainty(uncertainty_sens.array.T), 
            header=meta_sens['header'])

        nccd = CCDData(
            data=flux_obs.T, uncertainty=StdDevUncertainty(uncertainty_obs.array.T), 
            mask=nccd.mask, header=nccd.header)

    # Calibrate
    calibrated_ccd = flat_correct(ccd=nccd, flat=sccd, norm_value=1)

    # Output
    if nccd.uncertainty is None:
        calibrated_ccd.uncertainty = None

    if isinstance(ccd, CCDData):
        # Add headers here
        calibrated_ccd.header['EXPTIME'] = exptime.value
        calibrated_ccd.header['AIRMASS'] = sccd.header['AIRMASS']
        calibrated_ccd.header['CALIBRAT'] = '{} Calibrated'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        calibrated_ccd = np.ma.array(calibrated_ccd.data, mask=calibrated_ccd.mask)

    else:
        calibrated_ccd = deepcopy(calibrated_ccd.data)

    return calibrated_ccd
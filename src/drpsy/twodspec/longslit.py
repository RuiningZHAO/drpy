from copy import deepcopy
import warnings

# NumPy
import numpy as np
# SciPy
from scipy import interpolate, signal, ndimage
from scipy.optimize import OptimizeWarning
# matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.nddata import CCDData, StdDevUncertainty
# ccdproc
from ccdproc import flat_correct
# specutils
from specutils import Spectrum1D

from drpsy import conf
from drpsy.batch import CCDDataList
from drpsy.onedspec import loadExtinctionCurve
from drpsy.onedspec.center import _center1D_Gaussian, _refinePeakBases, _refinePeaks
from drpsy.modeling import Poly1D, Spline1D, Spline2D, GaussianSmoothing2D
from drpsy.plotting import plotFitting, _plot2d, _plotSpectrum1D
from drpsy.validate import (_validateBool, _validateString, _validateRange, 
                            _validateInteger, _validate1DArray, _validateNDArray, 
                            _validateCCDData, _validateCCDList, _validateCCD, 
                            _validateSpectrum, _validateBins, _validateAperture, 
                            _validatePath)

from .utils import invertCoordinateMap

# Set plot parameters
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['response', 'illumination', 'align', 'fitcoords', 'transform', 'trace', 
           'background', 'extract', 'calibrate2d']


def response(ccd, slit_along, n_piece=3, n_iter=5, sigma_lower=None, sigma_upper=None, 
             grow=False, use_mask=False, title='response', show=conf.show, 
             save=conf.save, path=conf.path):
    """Determine response calibration.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
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
        Response calibration frame, of the same type as the input ``ccd``.
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
    # Bad pixels (NaNs or infs) in the original frame (if any) may lead to unmasked 
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
        xlabel='dispersion axis [px]', ylabel='pixel value', title=title, 
        show=show, save=save, path=path, use_relative=True)

    if slit_along == 'col':
        rccd = CCDData(
            data=(np.tile(y_fit, (n_row, 1)) * nccd.unit), 
            uncertainty=StdDevUncertainty(np.full((n_row, n_col), uncertainty_y_fit)), 
            mask=np.tile(master_mask, (n_row, 1))
        )

    else:
        rccd = CCDData(
            data=(np.tile(y_fit, (n_row, 1)).T * nccd.unit), 
            uncertainty=StdDevUncertainty(np.full((n_col, n_row), uncertainty_y_fit)), 
            mask=np.tile(master_mask, (n_row, 1)).T
        )

    # Response calibrated frame
    nccd = flat_correct(ccd=nccd, flat=rccd, norm_value=1)

    if ccd.uncertainty is None:
        nccd.uncertainty = None

    if ccd.mask is None:
        nccd.mask = None

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['RESPONSE'] = '{} Response corrected.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd

# todo: improve multiplot (or plot arrays). doc.
def illumination(ccd, slit_along, method, sigma=None, n_piece=None, bins=5, n_iter=5, 
                 sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
                 title='illumination', show=conf.show, save=conf.save, path=conf.path):
    """Model illumination.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.

    method : str
        `Gaussian2D` or `CubicSpline2D` or `iraf`.
        - if `Gaussian2D`, ``sigma`` is required. [...]
        - if `CubicSpline2D`, ``n_piece`` is required. [...]
        - if `iraf`, ``n_piece`` is required. [...]

    sigma : scalar or sequence of scalars or `None`, optional
        Standard deviation for `~scipy.ndimage.gaussian_filter`.
        Default is `None`.

    n_piece : int or sequence of scalars or `None`, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        - if methed is `CubicSpline2D`, [...]
        - if method is `iraf`, [...]
    
    bins : int or sequence of scalars, optional
        If `int`, it defines the number of equal-width bins. If a sequence, it defines 
        a monotonically increasing array of bin edges, including the rightmost edge, 
        allowing for non-uniform bin widths.
        Default is `5`.
        
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
        - if method is `CubicSpline2D` or `iraf`, the masked pixels are ignored in the 
        fitting.
        
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
        # Both are `None` originally
        if n_iter == 0:
            bin_threshold_lower = [None] * n_bin
            bin_threshold_upper = [None] * n_bin
        # Both ``threshold_lower`` and ``threshold_upper`` are 2-dimensional arrays, 
        # though each array only has one row. Use flatten to get rid of the additional 
        # dimension.
        else:
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
            # Bad pixels (NaNs or infs) in the original frame (if any) may lead to 
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
            ylabel='pixel value', title=f'{title} at {slit_along} {loc_bin[i]}', 
            show=show, save=save, path=path, use_relative=True)
    
    # Illumination
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
    
    # NaNs and infs are interpolated.
    # Note that the ``use_mask`` flag is not used in this function.
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
              show=conf.show, save=conf.save, path=conf.path, **kwargs):
    """Fit distortion across the slit.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.
    
    order : int, optional
        Order of correction. Must be `2` >= ``order`` >= `0`.

    n_med : int, optional
        Number of pixels to median. Should be small for heavy distortion. Must be 
        positive.
        Default is `5`.
    
    prominence ：scalar or `~numpy.ndarray` or sequence, optional
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

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    # Apply mask
    data_arr[mask_arr] = np.nan

    _validateInteger(n_med, 'n_med', (1, None), (True, None))
    
    # Split into bins along spatial (slit) axis
    bin_edges = np.hstack([np.arange(0, n_row, n_med), n_row])
    bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T
    n_bin = bin_edges.shape[0]
    loc_bin = (bin_edges.sum(axis=1) - 1) / 2

    bin_data_arr = np.zeros((n_bin, n_col))

    # A precision of 0.05 px for cross-correlation (can be changed)
    n_sub = 20
    # Of type `~numpy.float64`, although still named as an index array.
    idx_col_dense = np.linspace(0, idx_col[-1], idx_col[-1] * n_sub + 1)
    bin_data_arr_dense = np.zeros((n_bin, idx_col_dense.shape[0]))

    for i, (bin_start, bin_end) in enumerate(bin_edges):
        # Bad pixels should be labeled by ``mask_arr`` in advance
        bin_data = np.nanmean(data_arr[bin_start:bin_end, :], axis=0)
        bin_mask = np.all(mask_arr[bin_start:bin_end, :], axis=0)

        # Interpolate masked pixels to ensure finite results in cross-correlation
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
    bin_data_ref_dense_ext = np.hstack(
        [bin_data_ref_dense[-n_ext * n_sub:], bin_data_ref_dense, 
         bin_data_ref_dense[:n_ext * n_sub]]
    )

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
        # frame. However, `~scipy.interpolate.LSQBivariateSpline` called by `Spline2D` 
        # does not support extrapolation. An alternative way is to use ``bbox``. Here 
        # we set ``bbox`` to be the boundary of the whole frame and therefore will get 
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

# todo: check Delaunay method for uncertainty estimation.
def transform(ccd, X, Y, flux=True):
    """Transform input frame to a new coordinate.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    X, Y : `~numpy.ndarray`
        X(U, V), Y(U, V)
    
    flux : bool, optional
        If `True` the interpolated output pixel value is multiplied by the Jacobean of 
        the transformation (essentially the ratio of pixel areas between the input and 
        output frames).
    
    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        The transformed frame.
    """

    nccd = _validateCCDData(ccd, 'ccd')

    data_arr = deepcopy(nccd.data)

    n_row, n_col = data_arr.shape

    idx_row, idx_col = np.arange(n_row), np.arange(n_col)

    _validateNDArray(X, 'X', 2)
    _validateNDArray(Y, 'Y', 2)

    if not (data_arr.shape == X.shape == Y.shape):
        raise ValueError(
            'The input frame and the coordinate arrays should have the same shape.')

    _validateBool(flux, 'flux')

    # Flux conservation consists of multiplying the interpolated pixel value by the 
    # Jacobean of the transformation at that point. This is essentially the ratio of 
    # the pixel areas between the input and output frames. (for details see 
    # https://iraf.net/irafhelp.php?val=longslit.transform&help=Help+Page)
    if flux:
        x = (idx_col[1:] + idx_col[:-1]) / 2
        a1 = interpolate.interp1d(
            x, np.diff(X, axis=1), axis=1, bounds_error=False, fill_value='extrapolate', 
            assume_sorted=True)(idx_col)
        y = (idx_row[1:] + idx_row[:-1]) / 2
        a2 = interpolate.interp1d(
            y, np.diff(Y, axis=0), axis=0, bounds_error=False, fill_value='extrapolate', 
            assume_sorted=True)(idx_row)
        S = a1 * a2

    else:
        S = 1.

    # Data
    data_arr_transformed = ndimage.map_coordinates(
        input=data_arr, coordinates=(Y, X), order=1, mode='nearest')
    data_arr_transformed *= S
    nccd.data = deepcopy(data_arr_transformed)

    # Uncertainty
    # This method is simple but enlarges the resulting uncertainty.
    if nccd.uncertainty is not None:
        uncertainty_arr = deepcopy(nccd.uncertainty.array)
        uncertainty_arr_transformed = np.sqrt(
            ndimage.map_coordinates(
                input=uncertainty_arr**2, coordinates=(Y, X), order=1, mode='nearest')
        )
        uncertainty_arr_transformed *= S
        nccd.uncertainty.array = deepcopy(uncertainty_arr_transformed)

    # Mask
    if nccd.mask is not None:
        mask_arr = nccd.mask.astype(float)
        mask_arr_transformed = ndimage.map_coordinates(
            input=mask_arr, coordinates=(Y, X), order=1, mode='constant', cval=1.)
        # Any interpolated value of 0.1 or greater is given the value 1 in the output 
        # mask. The choice of 0.1 is purely empirical and gives an approximate 
        # identification of significant affected pixels. (for details see 
        # https://iraf.net/irafhelp.php?val=longslit.transform&help=Help+Page)
        threshold = 0.1
        mask_arr_transformed[mask_arr_transformed >= threshold] = 1.
        mask_arr_transformed = np.round(mask_arr_transformed).astype(bool)
        nccd.mask = deepcopy(mask_arr_transformed)

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['TRANSFOR'] = '{} Transformed.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd

# todo: doc
def trace(ccd, slit_along, fwhm, method, interval=None, n_med=3, n_piece=3, n_iter=5, 
          sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
          title='trace', show=conf.show, save=conf.save, path=conf.path):
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

    # Bad pixels (NaNs or infs) in the original frame (if any) may lead to unmasked 
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

            # Bad pixels (NaNs or infs) in the original frame (if any) may lead to 
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
        # ax[0].set_yscale('log')
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

    # No uncertainty or mask frame
    trace1d = Spectrum1D(flux=(fitted_trace * u.pixel), meta=meta)

    return trace1d


def background(ccd, slit_along, trace1d, distance=50, aper_width=50, degree=1, 
               n_iter=5, sigma_lower=None, sigma_upper=None, grow=False, 
               use_mask=False, title='background', show=conf.show, save=conf.save, 
               path=conf.path):
    """Model background.
    
    Sky background of the input frame is modeled col by col (or row by row, depending 
    on the ``slit_along``) through polynomial fittings.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.
    
    trace1d : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray`
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
    if isinstance(trace1d, Spectrum1D):
        trace1d = trace1d.flux.value
    trace1d = _validate1DArray(trace1d, 'trace1d', n_col, True)

    distance = _validate1DArray(distance, 'distance', 2, True)

    aper_width = _validate1DArray(aper_width, 'aper_width', 2, True)
    for i in range(2):
        _validateRange(aper_width[i], f'aper_width[{i}]', (1, None), (True, None))

    # Usually there are two separate background apertures (though they can be set 
    # overlapped). ``distance`` controls separations on either side of the trace.
    idx_lbkg_min = trace1d - distance[0] - aper_width[0] / 2
    idx_lbkg_max = trace1d - distance[0] + aper_width[0] / 2
    idx_rbkg_min = trace1d + distance[1] - aper_width[1] / 2
    idx_rbkg_max = trace1d + distance[1] + aper_width[1] / 2
    
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
            ax[1].plot(idx_col + 1, trace1d + 1, 'r--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_lbkg_min + 1, 'y--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_lbkg_max + 1, 'y--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_rbkg_min + 1, 'b--', lw=1.5)
            ax[1].plot(idx_col + 1, idx_rbkg_max + 1, 'b--', lw=1.5)
        else:
            ax[1].plot(trace1d + 1, idx_row + 1, 'r--', lw=1.5)
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

    # Background frame
    if slit_along == 'col':
        nccd.data = deepcopy(bkg_arr)

        if nccd.uncertainty is not None:
            nccd.uncertainty.array = deepcopy(std_arr)

        if nccd.mask is not None:
            nccd.mask = deepcopy(mask_arr)
    else:
        nccd.data = bkg_arr.T

        if nccd.uncertainty is not None:
            nccd.uncertainty.array = std_arr.T

        if nccd.mask is not None:
            nccd.mask = mask_arr.T

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKBACKGR'] = '{} Background.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = deepcopy(nccd.data)

    return nccd


def extract(ccd, slit_along, trace1d, aper_width, n_aper=1, spectral_axis=None, 
            title='aperture', show=conf.show, save=conf.save, path=conf.path):
    """Extract 1-dimensional spectra.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.
    
    trace1d : `~specutils.Spectrum1D` or scalar or `~numpy.ndarray`
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
    if isinstance(trace1d, Spectrum1D):
        trace1d = trace1d.flux.value
    trace1d = _validate1DArray(trace1d, 'trace1d', n_col, True)

    aper_width = _validateAperture(aper_width)
    
    _validateInteger(n_aper, 'n_aper', (1, None), (True, None))

    # The total aperture width is the sum of ``aper_width[0]`` and ``aper_width[1]``. 
    # If they have opposite signs, the whole aperture will be on one side of the trace.
    aper_edges = trace1d + np.linspace(
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
                data_aper[j, i] += (
                    data_arr[idx_end, i] * (aper_end[j] - aper_start[j]))
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

    if n_aper == 1:
        spectrum1d = spectrum1d[0]

    return spectrum1d


def calibrate2d(ccd, slit_along, exptime, airmass, extinct, sens1d, 
                use_uncertainty=False):
    """Apply a flux calibration to a 2-dimensional spectrum.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
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

    _validateBool(use_uncertainty, 'use_uncertainty')

    new_sens1d, _, sens, uncertainty_sens, _ = _validateSpectrum(
        sens1d, 'sens1d', use_uncertainty, False)

    # ``sens``, ``uncertainty_sens`` and ``mask_sens`` should be 1-dimensional arrays. 
    # Use flatten to get rid of additional dimensions.
    if sens.ndim > 1:
        sens = sens.flatten()[:sens.shape[-1]]
        uncertainty_sens = uncertainty_sens.flatten()[:uncertainty_sens.shape[-1]]

    wavelength_sens = new_sens1d.wavelength.value

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

    sens = 10**(0.4 * sens)
    uncertainty_sens *= 0.4 * np.log(10) * sens
    
    _validateString(slit_along, 'slit_along', ['col', 'row'])

    if slit_along == 'col':
        nccd, data_arr, _, _ = _validateCCD(ccd, 'ccd', False, False, False)

    else:
        nccd, data_arr, _, _ = _validateCCD(ccd, 'ccd', False, False, True)

    n_row, n_col = data_arr.shape

    if wavelength_sens.shape[0] != n_col:
        raise ValueError(
            'The spectral axes of ``ccd`` and ``sens1d`` should be the same.')

    if isinstance(exptime, str):
        exptime = nccd.header[exptime] * u.s

    bin_width = np.abs(np.diff(new_sens1d.bin_edges.to(u.AA)))

    flux_obs = (data_arr * nccd.unit) / (exptime * bin_width) # [counts/s/Angstrom]

    if nccd.uncertainty is not None:
        uncertainty_obs = nccd.uncertainty.array / (exptime.value * bin_width.value)

    else:
        uncertainty_obs = None

    sens = np.tile(sens, (n_row, 1)) * new_sens1d.flux.unit

    uncertainty_sens = np.tile(uncertainty_sens, (n_row, 1))

    if 'header' in new_sens1d.meta:
        header_sens = new_sens1d.meta['header']

    else:
        header_sens = None

    if slit_along == 'col':
        sccd = CCDData(
            data=sens, uncertainty=StdDevUncertainty(uncertainty_sens), 
            header=header_sens)
        nccd = CCDData(
            data=flux_obs, uncertainty=StdDevUncertainty(uncertainty_obs), 
            mask=nccd.mask, header=nccd.header)

    else:
        sccd = CCDData(
            data=sens.T, uncertainty=StdDevUncertainty(uncertainty_sens.T), 
            header=header_sens)
        nccd = CCDData(
            data=flux_obs.T, uncertainty=StdDevUncertainty(uncertainty_obs.T), 
            mask=nccd.mask, header=nccd.header)

    # Calibrate
    calibrated_ccd = flat_correct(ccd=nccd, flat=sccd, norm_value=1)

    # Output
    if nccd.uncertainty is None:
        calibrated_ccd.uncertainty = None

    if isinstance(ccd, CCDData):
        # Add headers here
        calibrated_ccd.header['EXPTIME'] = exptime.value
        calibrated_ccd.header['AIRMASS'] = airmass
        calibrated_ccd.header['CALIBRAT'] = '{} Calibrated'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        calibrated_ccd = np.ma.array(calibrated_ccd.data, mask=calibrated_ccd.mask)

    else:
        calibrated_ccd = deepcopy(calibrated_ccd.data)

    return calibrated_ccd
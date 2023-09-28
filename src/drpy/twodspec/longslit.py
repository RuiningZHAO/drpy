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
from astropy.convolution import convolve_fft, Gaussian2DKernel, interpolate_replace_nans
from astropy.utils.exceptions import AstropyUserWarning
# ccdproc
from ccdproc import flat_correct
# specutils
from specutils import Spectrum1D
# drpy
from drpy import conf
from drpy.batch import CCDDataList
from drpy.onedspec import loadExtinctionCurve
from drpy.onedspec.center import _center1D_Gaussian, _refinePeakBases, _refinePeaks
from drpy.modeling import Poly1D, Spline1D, Spline2D, GaussianSmoothing2D
from drpy.plotting import plotFitting, _plotFitting, plot2d, _plot2d
from drpy.validate import (_validateBool, _validateString, _validateRange, 
                           _validateInteger, _validate1DArray, _validateNDArray, 
                           _validateCCDData, _validateCCDList, _validateCCD, 
                           _validateSpectrum, _validateBins, _validateAperture, 
                           _validatePath)
from drpy.decorate import ignoreWarning

from .utils import invertCoordinateMap

sigma_clip = ignoreWarning(sigma_clip, 'ignore', AstropyUserWarning)

# Set plot parameters
plt.rcParams['figure.figsize'] = [conf.fig_width, conf.fig_width]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['response', 'illumination', 'align', 'fitcoords', 'transform', 'trace', 
           'background', 'profile', 'extract', 'calibrate2d']


def response(ccd, slit_along, n_piece=3, coordinate=None, maxiters=5, sigma_lower=None, 
             sigma_upper=None, grow=False, use_mask=False, plot=True, 
             path=conf.fig_path):
    """Model response.

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

    coordinate : `~numpy.ndarray` or `None`, optional
        One of the coordinate maps describing the curvature of the input frame. A 
        one-to-one mapping is assumed for the other.
        Default is `None`.

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
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the fitting. 
        Default is `False`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Modeled response, of the same type as the input ``ccd``. The mask attached (if 
        any) is for the input frame rather than the modeled response.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)

        if coordinate is not None:
            coordinate = _validateNDArray(coordinate, 'coordinate', 2)

    else:
        nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, True)

        if coordinate is not None:
            coordinate = _validateNDArray(coordinate, 'coordinate', 2).T

    n_row, n_col = data_arr.shape

    # Apply mask
    data_arr[mask_arr] = np.nan

    # Average along spatial (slit) axis
    x = np.arange(n_col)
    # Bad pixels (NaNs or infs) in the original frame (if any) may lead to unmasked 
    # elements in ``y`` and may cause an error in the spline fitting below.
    y = np.nanmean(data_arr, axis=0)
    m_y = np.all(mask_arr, axis=0)

    # Fit cubic spline function (always float64)
    spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
        x=x, y=y, m=m_y, order=3, n_piece=n_piece, maxiters=maxiters,  
        sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, use_relative=True)

    # Control precision
    y_fit = spl(x).astype(ccd.dtype)

    # Fitting plot
    plotFitting(
        x=x, y=y, m=master_mask, y_fit=y_fit, r=residual, 
        threshold_lower=threshold_lower, threshold_upper=threshold_upper, 
        xlabel='dispersion axis [px]', ylabel='pixel value', title='response', 
        show=False, save=plot, path=path, use_relative=True)

    if coordinate is not None:
        data_fit = spl(coordinate).astype(ccd.dtype)
    else:
        data_fit = np.tile(y_fit, (n_row, 1))

    if slit_along == 'col':
        nccd.data = data_fit.copy()
        if ccd.mask is not None:
            nccd.mask = np.tile(master_mask, (n_row, 1))
    else:
        nccd.data = data_fit.T
        if ccd.mask is not None:
            nccd.mask = np.tile(master_mask, (n_row, 1)).T

    nccd.uncertainty = None

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['RESPONSE'] = '{} Response.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return nccd

# todo: doc.
def illumination(ccd, slit_along, method, sigma=None, n_piece=None, bins=5, maxiters=5, 
                 sigma_lower=None, sigma_upper=None, grow=False, use_mask=False, 
                 plot=conf.fig_save, path=conf.fig_path):
    """Model illumination. Uncertainty of the model may be dominated by systematic 
    error which cannot be estimated from data.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.

    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned frame, which is 
        always of the same shape as ``ccd``.

    method : {'Gaussian2D', 'CubicSpline2D', 'iraf'}
        Method to model illumination.
        - if `Gaussian2D`, ``sigma`` is required. [...]
        - if `CubicSpline2D`, both ``sigma`` and ``n_piece`` are required. [...]
        - if `iraf`, ``n_piece`` is required. [...]

    sigma : scalar or sequence of scalars or `None`, optional
        Standard deviation for Gaussian filter. If sequence of scalars, the first 
        element should be standard deviation along the x axis.
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
    # row), i.e., data_arr[:, bin_edges[-1][0]:bin_edges[-1][1]].
    if slit_along == 'col':
        bin_edges, loc_bin, n_bin = _validateBins(bins, n_col)
    else:
        bin_edges, loc_bin, n_bin = _validateBins(bins, n_row)
    
    _validateString(method, 'method', ['Gaussian2D', 'CubicSpline2D', 'iraf'])

    if '2D' in method: # 2-dimensional method (`Gaussian2D` or `CubicSpline2D`)
        
        if slit_along == 'col':

            try:
                sigma = sigma[::-1]

            except:
                pass
        
        else:
            try:
                n_piece = n_piece[::-1]

            except:
                pass
        
        # (X, Y) grid
        X = np.tile(idx_col, (n_row, 1))
        Y = np.tile(idx_row, (n_col, 1)).T
        
        # Interpolate masked pixels
        if mask_arr.any():
            
            # 2-dimensional Gaussian kernel
            x_stddev, y_stddev = sigma
            x_size, y_size = int(8 * x_stddev + 1), int(8 * y_stddev + 1)
            kernel = Gaussian2DKernel(
                x_stddev=x_stddev, y_stddev=y_stddev, theta=0.0, x_size=x_size, 
                y_size=y_size, mode='center')
            
            # `~astropy.convolution.convolve_fft` is faster in most cases. However, it 
            # does not support `extend` method and may lead to bad boundary values. 
            # Therefore, the original frame is extended before convolution by half of 
            # the kernal size. The padding pixels are filled with nearest good values.
            
            # Extend (X, Y) grid
            x_pad, y_pad = (x_size - 1) // 2, (y_size - 1) // 2
            idx_row_extended = np.arange(idx_row[0] - y_pad, idx_row[-1] + y_pad + 1)
            idx_col_extended = np.arange(idx_col[0] - x_pad, idx_col[-1] + x_pad + 1)
            X_extended = np.tile(idx_col_extended, (n_row + y_size - 1, 1))
            Y_extended = np.tile(idx_row_extended, (n_col + x_size - 1, 1)).T
            
            # Extend data array
            data_arr_extended = interpolate.griddata(
                points=(X[~mask_arr], Y[~mask_arr]), values=data_arr[~mask_arr], 
                xi=(X_extended.flatten(), Y_extended.flatten()), method='nearest'
            ).reshape(X_extended.shape)
            
            # Apply mask
            data_arr_extended[y_pad:-y_pad, x_pad:-x_pad][mask_arr] = np.nan
            
            # Replace NaNs
            data_arr = interpolate_replace_nans(
                data_arr_extended, kernel, boundary='fill', fill_value=0., 
                convolve=convolve_fft)[y_pad:-y_pad, x_pad:-x_pad]

        if method == 'Gaussian2D':

            data_fit, residual, threshold_lower, threshold_upper, master_mask = (
                GaussianSmoothing2D(
                    X, Y, data_arr, sigma, maxiters=maxiters, sigma_lower=sigma_lower, 
                    sigma_upper=sigma_upper, axis=0, grow=grow, use_relative=True)
            )

        elif method == 'CubicSpline2D':
            
            bispl, residual, threshold_lower, threshold_upper, master_mask = Spline2D(
                x=X, y=Y, z=data_arr, order=(3, 3), n_piece=n_piece, maxiters=maxiters, 
                sigma_lower=sigma_lower, sigma_upper=sigma_upper, axis=0, grow=grow, 
                use_relative=True)

            data_fit = bispl(idx_col, idx_row, grid=True).T
        
        master_mask |= mask_arr
        
        data_fit = data_fit.astype(ccd.dtype)

        # In 2D case, ``loc_bin`` is only used as index, thus converted to ``int``.
        idx_bin = loc_bin.astype(int)
        bin_data_arr = data_arr[:, idx_bin].T
        bin_data_fit = data_fit[:, idx_bin].T
        bin_mask_arr = master_mask[:, idx_bin].T
        bin_residual = residual[:, idx_bin].T
        # Both are `None` originally
        if maxiters == 0:
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
                x=idx_row, y=bin_data_arr[i], m=bin_mask, order=3, n_piece=n_piece[i], 
                maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                grow=grow, use_relative=True)
            bin_data_fit[i] = bin_spl(idx_row)
            master_mask[bin_mask_arr[i], bin_start:bin_end] = True
        
        # Interpolate
        data_fit = interpolate.interp1d(
            x=loc_bin, y=bin_data_fit.T, axis=1, kind='linear', bounds_error=False, 
            fill_value='extrapolate', assume_sorted=True)(idx_col).astype(ccd.dtype)

    if slit_along != 'col':
        data_fit = data_fit.T
        master_mask = master_mask.T
        n_col, n_row = n_row, n_col
        idx_col, idx_row = idx_row, idx_col

    # Fitting plot
    _validateBool(plot, 'plot')

    if plot:

        if slit_along == 'col':
            x = idx_row
        else:
            x = idx_col

        fig_path = _validatePath(path, 'illumination fitting', '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in range(n_bin):

                fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                fig.subplots_adjust(hspace=0)

                _plotFitting(
                    ax=ax, x=x, y=bin_data_arr[i], m=bin_mask_arr[i], x_fit=x, 
                    y_fit=bin_data_fit[i], r=bin_residual[i], 
                    threshold_lower=bin_threshold_lower[i], 
                    threshold_upper=bin_threshold_upper[i], xlabel='spatial axis [px]', 
                    ylabel='pixel value', use_relative=True)

                ax[0].set_title(f'profile at column {loc_bin[i]}', fontsize=16)
                fig.align_ylabels()
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()
    
    # Illumination
    nccd.data = data_fit.copy()
    nccd.uncertainty = None
    if nccd.mask is not None:
        nccd.mask[master_mask] = True

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['MKILLUM'] = '{} Illumination.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

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


def fitcoords(ccd, slit_along, order=0, n_med=5, prominence=1e-3, n_piece=3, 
              maxiters=5, sigma_lower=None, sigma_upper=None, grow=False, 
              use_mask=False, plot=conf.fig_save, path=conf.fig_path, **kwargs):
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
    
    # Split into bins along spatial (slit) axis
    _validateInteger(n_med, 'n_med', (1, None), (True, None))
    bin_edges, loc_bin, n_bin = _validateBins(n_med, n_row, isWidth=True)

    idx_bin = np.arange(n_bin)

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

    for i in idx_bin:
        # Get zeropoint shift through cross-correlation
        idx_max = signal.correlate(
            bin_data_ref_dense_ext, bin_data_arr_dense[i], mode='valid', method='auto'
        ).argmax()
        shift_arr[i] = idx_ref_dense_ext[idx_max]

    _validateInteger(order, 'order', (0, 2), (True, True))

    if order == 0:

        # Univariate cubic spline fitting
        spl, residual, threshold_lower, threshold_upper, mask = Spline1D(
            x=loc_bin, y=shift_arr, order=3, n_piece=n_piece, maxiters=maxiters, 
            sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)
        shift_fit = spl(idx_row)

        U = idx_col + shift_fit[:, np.newaxis]

        # Fitting plot
        plotFitting(
            x=loc_bin, y=shift_arr, m=mask, x_fit=idx_row, y_fit=shift_fit, r=residual, 
            threshold_lower=threshold_lower, threshold_upper=threshold_upper, 
            xlabel='spatial axis [px]', ylabel='shift value [px]', 
            title='zeropoint shift curve', show=False, save=plot, path=path, 
            use_relative=False)

    else:

        shift = np.round(shift_arr * n_sub).astype(int)
        rolled_bin_data_arr_dense = np.zeros_like(bin_data_arr_dense)
        for i in idx_bin:
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

        for i in idx_bin:

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
        x = refined_peaks_arr.copy()
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
            x=x, y=y, z=z, m=mask, order=(order, 3), n_piece=(1, n_piece), bbox=bbox, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, axis=None, 
            grow=grow, use_relative=False)

        # !!! Extrapolation is used here (see above) !!!
        U = bispl(idx_col, idx_row, grid=True).T

        # Plot
        _validateBool(plot, 'plot')

        if plot:
            
            # Peak detection plot
            title = 'fitcoords peak detection'
            
            # Split into subplots
            n_subplot = 2

            fig, ax = plt.subplots(n_subplot, 1, dpi=100)

            length = n_col // n_subplot + 1
            for i in range(n_subplot):

                idx_start, idx_end = i * length, (i + 1) * length
                idx_peak = np.where((idx_start <= peaks) & (peaks < idx_end))[0]

                ax[i].step(
                    idx_col[idx_start:idx_end], bin_data_mean[idx_start:idx_end], 
                    color='k', ls='-', where='mid')
                for idx in idx_peak:
                    ymin = heights[idx] * 1.2
                    ymax = heights[idx] * 1.5
                    ax[i].plot([peaks[idx], peaks[idx]], [ymin, ymax], 'r-', lw=1.5)

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
            ax[0].set_title(title, fontsize=16)
            fig.align_ylabels()
            fig.set_figheight(fig.get_figwidth() * n_subplot / 2)
            fig.tight_layout()

            # Save
            fig_path = _validatePath(path, title)
            plt.savefig(fig_path, dpi=100)

            plt.close()

            # Distortion fitting plot
            z_fit = bispl(idx_col, loc_bin, grid=True).T

            fig_path = _validatePath(path, 'distortion fitting', '.pdf')
            with PdfPages(fig_path, keep_empty=False) as pdf:

                for i in idx_bin[::10]:

                    fig, ax = plt.subplots(
                        2, 1, sharex=True, height_ratios=[3, 1], dpi=100)
                    fig.subplots_adjust(hspace=0)

                    _plotFitting(
                        ax=ax, x=refined_peaks_arr[i], 
                        y=(peaks - refined_peaks_arr[i]), m=master_mask[i], 
                        x_fit=idx_col, y_fit=(z_fit[i] - idx_col), r=residual[i], 
                        threshold_lower=threshold_lower, 
                        threshold_upper=threshold_upper, xlabel='dispersion axis [px]', 
                        ylabel='shift [px]', use_relative=False)

                    ax[0].set_title(f'distortion at bin {loc_bin[i]}', fontsize=16)
                    fig.align_ylabels()
                    fig.tight_layout()

                    pdf.savefig(fig, dpi=100)

                    plt.close()

        # Distortion residual image
        residual[master_mask] = np.nan
        cmap = plt.cm.get_cmap('Greys_r').copy(); cmap.set_bad('red', 1.)
        plot2d(
            residual, cmap=cmap, xlabel='peak number', ylabel='bin number', 
            cblabel='pixel', title='distortion residual', show=False, save=plot, 
            path=path)

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

    nccd, data_arr, _, _ = _validateCCD(ccd, 'ccd', False, False, False)

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
    nccd.data = data_arr_transformed.copy()

    # Uncertainty
    # This method is simple but enlarges the resulting uncertainty.
    if nccd.uncertainty is not None:
        uncertainty_arr = nccd.uncertainty.array.copy()
        uncertainty_arr_transformed = np.sqrt(
            ndimage.map_coordinates(
                input=uncertainty_arr**2, coordinates=(Y, X), order=1, mode='nearest')
        )
        uncertainty_arr_transformed *= S
        nccd.uncertainty.array = uncertainty_arr_transformed.copy()

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
        nccd.mask = mask_arr_transformed.copy()

    # Output
    if isinstance(ccd, CCDData):
        nccd.header['TRANSFOR'] = '{} Transformed.'.format(
            Time.now().to_value('iso', subfmt='date_hm'))

    elif np.ma.isMaskedArray(ccd):
        nccd = np.ma.array(nccd.data, mask=nccd.mask)

    else:
        nccd = nccd.data.copy()

    return nccd

# todo: doc
def trace(ccd, slit_along, fwhm, method, n_med=3, reference_bin=None, interval=None, 
          n_piece=3, maxiters=5, sigma_lower=None, sigma_upper=None, grow=False, 
          negative=False, use_mask=False, title='trace', show=conf.fig_show, 
          save=conf.fig_save, path=conf.fig_path):
    """Trace on the 2-dimensional spectrum.
    
    First the spatial profiles are binned by taking median along the dispersion axis. 
    Then the center of the specified feature (that is the strongest one in the 
    interval) in the reference bin is determined by fitting a Gaussian profile. [...]
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    slit_along : str
        `col` or `row`. If `row`, the data array of ``ccd`` will be transposed during 
        calculations. Note that this will NOT affect the returned trace.

    fwhm : scalar
        Estimated full width at half maximum of the peak to be traced. (Rough 
        estimation is enough.)

    method : str
        `center` or `trace`, and
        - if `center`, a straight trace at the center of the specified feature in the 
        middle bin is returned.
        - if `trace`, the center of the specified feature is treated as an initial
        guess for bin by bin Gaussian fittings.
    
    n_med : int, optional
        Number of spatial profiles to median. Must be >= `3`. Large number for faint 
        source.
        Default is `3`.
    
    reference_bin : int or `None`, optional
        Index of the reference bin.
        If `None`, the reference bin is the middle bin.
    
    interval : 2-tuple or `None`, optional
        Interval the specified feature lies in, and
        - if `None`, the brightest source in the frame is traced.
        - if 2-tuple, the brightest source in the interval is traced. (Use this when 
        the source is not the brightest one in the frame)
    
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
    
    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(negative, 'negative')

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
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
    
    # Reference bin
    ref_edge = bin_edges[reference_bin]
    
    # Bad pixels (NaNs or infs) in the original frame (if any) may lead to 
    # unmasked elements in ``count_ref`` and may cause an error in the 
    # Gaussian fitting below.
    count_ref = np.nanmedian(data_arr[:, ref_edge[0]:ref_edge[1]], axis=1)
    mask_ref = np.all(mask_arr[:, ref_edge[0]:ref_edge[1]], axis=1)
    
    if interval is None:
        interval = (0, None)
    else:
        interval = _validate1DArray(interval, 'interval', 2, False)
    
    # Maximum location (should be close to peak center)
    idx_src_ref = interval[0] + np.ma.argmax(
        np.ma.array(count_ref, mask=mask_ref)[interval[0]:interval[1]], 
        fill_value=-np.inf)
    
    # Peak range
    idx_min_ref = int(idx_src_ref - 0.5 * fwhm)
    idx_max_ref = int(idx_src_ref + 0.5 * fwhm)
    
    # Initial guess for Gaussian fitting   
    initial_guess = (1, idx_src_ref, fwhm)
    
    x_ref = idx_min_ref + np.arange(idx_max_ref + 1 - idx_min_ref)
    # Normalize
    y_ref = count_ref[idx_min_ref:(idx_max_ref + 1)] / count_ref[idx_src_ref]
    m_ref = mask_ref[idx_min_ref:(idx_max_ref + 1)]
    
    # Gaussian fitting
    try:
        center_ref = _center1D_Gaussian(x_ref[~m_ref], y_ref[~m_ref], initial_guess, 0)
    
    except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
        raise RuntimeError('No trace found in the given interval.')
    
    _validateString(method, 'method', ['center', 'trace'])
    
    if method == 'center':
    
        fitted_trace = np.full(n_col, center_ref)
    
    else:
        
        refined_trace = np.full(n_bin, np.nan)
        refined_trace[reference_bin] = center_ref

        for i in range(reference_bin + 1, n_bin):

            # Bad pixels (NaNs or infs) in the original frame (if any) may lead to 
            # unmasked elements in ``count_bin`` and may cause an error in the 
            # Gaussian fitting below.
            count_bin = np.nanmedian(
                data_arr[:, bin_edges[i][0]:bin_edges[i][1]], axis=1)
            mask_bin = np.all(mask_arr[:, bin_edges[i][0]:bin_edges[i][1]], axis=1)

            # Peak range
            idx_ref = np.where(~np.isnan(refined_trace))[0].max()
            idx_min_bin = int(refined_trace[idx_ref] - 0.5 * fwhm)
            idx_max_bin = int(refined_trace[idx_ref] + 0.5 * fwhm)

            # Peak center
            idx_src_bin = idx_min_bin + np.ma.argmax(
                np.ma.array(count_bin, mask=mask_bin)[idx_min_bin:idx_max_bin], 
                fill_value=-np.inf)

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
                continue

        for i in range(reference_bin - 1, -1, -1):

            # Bad pixels (NaNs or infs) in the original frame (if any) may lead to 
            # unmasked elements in ``count_bin`` and may cause an error in the 
            # Gaussian fitting below.
            count_bin = np.nanmedian(
                data_arr[:, bin_edges[i][0]:bin_edges[i][1]], axis=1)
            mask_bin = np.all(mask_arr[:, bin_edges[i][0]:bin_edges[i][1]], axis=1)

            # Peak range
            idx_ref = np.where(~np.isnan(refined_trace))[0].min()
            idx_min_bin = int(refined_trace[idx_ref] - 0.5 * fwhm)
            idx_max_bin = int(refined_trace[idx_ref] + 0.5 * fwhm)

            # Peak center
            idx_src_bin = idx_min_bin + np.ma.argmax(
                np.ma.array(count_bin, mask=mask_bin)[idx_min_bin:idx_max_bin], 
                fill_value=-np.inf)

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
                continue

        mask = np.isnan(refined_trace)
        
        # Spline fitting
        spl, residual, threshold_lower, threshold_upper, master_mask = Spline1D(
            x=loc_bin, y=refined_trace, m=mask, order=3, n_piece=n_piece, 
            maxiters=maxiters, sigma_lower=sigma_lower, sigma_upper=sigma_upper, grow=grow, 
            use_relative=False)

        fitted_trace = spl(idx_col)
    
    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'trace':
            title = f'{title} trace'

        # Trace fitting plot
        if method == 'trace':
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
        ax[0].step(idx_row, count_ref, 'k-', lw=1.5, where='mid')
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
        if slit_along == 'col':
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
    header['TRINTERV'] = f'{interval}'
    header['TRCENTER'] = center_ref
    header['TRACE'] = '{} Trace ({}, n_med = {}, n_piece = {})'.format(
        Time.now().to_value('iso', subfmt='date_hm'), method, n_med, n_piece)
    
    meta = {'header': header}

    # No uncertainty or mask frame
    trace1d = Spectrum1D(flux=(fitted_trace * u.pixel), meta=meta)

    return trace1d


def background(ccd, slit_along, trace1d=None, location=75, aper_width=50, degree=0, 
               maxiters=5, sigma_lower=None, sigma_upper=None, grow=False, 
               use_uncertainty=False, use_mask=False, title='background', 
               show=conf.fig_show, save=conf.fig_save, path=conf.fig_path):
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

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(use_uncertainty, 'use_uncertainty')

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
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
        if slit_along == 'col':
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
    if slit_along == 'col':

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


def profile(ccd, slit_along, trace1d, profile_width, window_length, polyorder=3, 
            deriv=0, delta=1.0, title='profile', show=conf.fig_show, 
            save=conf.fig_save, path=conf.fig_path):
    """Build an effective spatial profile along slit. Usually used by the optimal 
    extraction algorithm.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame. Should be background subtracted.

    slit_along : str
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

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    _validateBool(use_mask, 'use_mask')

    if slit_along == 'col':
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
        if slit_along == 'col':
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


def extract(ccd, slit_along, method, trace1d=None, aper_width=None, n_aper=None, 
            profile2d=None, background2d=None, rdnoise=None, maxiters=None, 
            sigma_lower=None, sigma_upper=None, grow=None, spectral_axis=None, 
            use_uncertainty=True, use_mask=True, title='aperture', show=conf.fig_show, 
            save=conf.fig_save, path=conf.fig_path):
    """Extract 1-dimensional spectra.
    
    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input frame.
    
    slit_along : str
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

    _validateString(slit_along, 'slit_along', ['col', 'row'])
    
    if slit_along == 'col':
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
        if slit_along == 'col':
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
            airmass = ccd.meta[airmass]

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
        calibrated_ccd = calibrated_ccd.data.copy()

    return calibrated_ccd
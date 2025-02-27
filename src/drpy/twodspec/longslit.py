from copy import deepcopy

# NumPy
import numpy as np
# SciPy
from scipy import interpolate, signal, ndimage
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.convolution import convolve_fft, Gaussian2DKernel, interpolate_replace_nans
# ccdproc
from ccdproc import flat_correct
from ccdproc.utils.slices import slice_from_string
# specutils
from specutils import Spectrum1D
# drpy
from drpy import conf
from drpy.batch import CCDDataList
from drpy.onedspec import loadExtinctionCurve
from drpy.onedspec.center import _refinePeakBases, _refinePeaks
from drpy.modeling import Spline1D, Spline2D, GaussianSmoothing2D
from drpy.plotting import plotFitting, _plotFitting, plot2d
from drpy.validate import (_validateBool, _validateString, _validateInteger, 
                           _validate1DArray, _validateNDArray, _validateCCDList, 
                           _validateCCD, _validateSpectrum, _validateBins, 
                           _validatePath)

# Set plot parameters
plt.rcParams['figure.figsize'] = [conf.fig_width, conf.fig_width]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['response', 'illumination', 'align', 'fitcoords', 'transform', 'calibrate2d']


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


def align(ccdlist, slit_along, index=0, interval='[:]'):
    """Align multiple frames.
    
    The median spatial profile of each frame is derived, followed by cross-correlation 
    between the the reference frame and the others along spatial axis. The minimum step 
    for cross-correlation is 1 [px].

    Parameters
    ----------
    ccdlist : Iterable
        Iterable object containing at least two 2-dimensional frames.

    slit_along : str
        `col` or `row`.
    
    index : int, optional
        Index of the reference frame.

    interval : str, optional
        Interval of the median spatial profile used for cross-correlation. If the whole 
        profile is used, there will be a risk (mostly for moving targets) that the 
        spectrum of the target might be aligned to the spectrum of a bright field star 
        passing-by. (Use this when there is a spectrum of a bright field star in the 
        frame)
        Default is `[:]`.

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

    slice_interval = slice_from_string(interval)
    
    # Median is used (instead of mean) to get rid of cosmic rays. This allows customers
    # to remove cosmic rays (after alignment) through frame combination. ``a`` does not 
    # contain any NaNs or infs in most cases, unless there are bad columns along 
    # dispersion axis.
    a = np.nanmedian(ccd_ref.data, axis=axis)

    x = np.arange(a.shape[0])
    
    # NaNs and infs are interpolated.
    # Note that the ``use_mask`` flag is not used in this function.
    isFinite = np.isfinite(a[slice_interval])
    fill_value = (a[slice_interval][isFinite][0], a[slice_interval][isFinite][-1])
    a = interpolate.interp1d(
        x=x[slice_interval][isFinite], y=a[slice_interval][isFinite], kind='linear', 
        bounds_error=False, fill_value=fill_value, assume_sorted=True)(x)

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
        isFinite = np.isfinite(v[slice_interval])
        fill_value = (v[slice_interval][isFinite][0], v[slice_interval][isFinite][-1])
        v = interpolate.interp1d(
            x=x[slice_interval][isFinite], y=v[slice_interval][isFinite], 
            kind='linear', bounds_error=False, fill_value=fill_value, 
            assume_sorted=True)(x)

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

        # Bad pixels should be labeled by ``mask_arr`` in advance. Median is used here, 
        # as it is more robust than mean when cosmic ray pixels exist.
        bin_data = np.nanmedian(data_arr[bin_start:bin_end, :], axis=0)
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

        # Update number of peaks
        n_peak = len(refined_peaks)

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

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    if slit_along == 'col':
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', False, False, False)

    else:
        nccd, data_arr, uncertainty_arr, mask_arr = _validateCCD(
            ccd, 'ccd', False, False, True)

    n_row, n_col = data_arr.shape

    new_sens1d, _, sens, uncertainty_sens, _ = _validateSpectrum(
        sens1d, 'sens1d', use_uncertainty, False)

    _validateBool(use_uncertainty, 'use_uncertainty')

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

    sens = np.tile(sens, (n_row, 1)) * new_sens1d.flux.unit
    uncertainty_sens = StdDevUncertainty(np.tile(uncertainty_sens, (n_row, 1)))

    if 'header' in new_sens1d.meta:
        header_sens = new_sens1d.meta['header']
    else:
        header_sens = None

    sccd = CCDData(data=sens, uncertainty=uncertainty_sens, header=header_sens)

    if wavelength_sens.shape[0] != n_col:
        raise ValueError(
            'The spectral axes of ``ccd`` and ``sens1d`` should be the same.')

    if isinstance(exptime, str):
        exptime = nccd.header[exptime] * u.s

    bin_width = np.abs(np.diff(new_sens1d.bin_edges.to(u.AA)))

    flux_obs = (data_arr * nccd.unit) / (exptime * bin_width) # [counts/s/Angstrom]

    uncertainty_obs = StdDevUncertainty(
        uncertainty_arr / (exptime.value * bin_width.value))

    nccd = CCDData(
        data=flux_obs, uncertainty=uncertainty_obs, mask=mask_arr, header=nccd.header)

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
# NumPy
import numpy as np
# Astropy
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
# SciPy
from scipy import interpolate
from scipy.ndimage import gaussian_filter
# drpy
from drpy.decorate import filterWarning
from drpy.validate import _validateBool, _validateInteger, _validate1DArray

sigma_clip = filterWarning('ignore', AstropyUserWarning)(sigma_clip)

__all__ = ['Poly1D', 'Spline1D', 'Spline2D', 'GaussianSmoothing2D']


def Poly1D(x, y, w=None, m=None, deg=1, maxiters=5, sigma_lower=None, sigma_upper=None, 
           grow=False, use_relative=False):
    """Fit 1-dimensional polynomial function.
    
    Sigma clipping is used to mask bad data points.

    Parameters
    ----------
    x : array_like
        Input dimension of data points – must be increasing.

    y : array_like
        Input dimension of data points.

    w : array_like, optional
        Weights for polynomial fitting. Must be positive. If `None` (default), weights 
        are all equal.

    m : array_like, optional
        Initial mask for polynomial fitting. If `None` (default), data points are all 
        unmasked.

    deg : int, optional
        Degree of the fitting polynomial.
        Default is `1`.

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

    use_relative : bool, optional
        If `True`, relative residual is used in sigma slipping instead of residual. 
        Default is `False`.

    Returns
    -------
    p : callable
        Callable object returned by `~numpy.poly1d`.

    residual : `~numpy.ndarray`
        Residual or relative residual, depending on ``use_relative``.

    threshold_lower : scalar or `None`
        Lower threshold for clipping.

    threshold_upper : scalar or `None`
        Upper threshold for clipping.

    master_mask : `~numpy.ndarray`
        Final mask after clipping.
    """

    if w is None:
        w = np.ones_like(y)

    if m is None:
        m = np.zeros_like(y, dtype=bool)

    _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

    _validateBool(use_relative, 'use_relative')

    # Polynomial fitting iteratively
    mask_new = m.copy()
    mask_old = np.ones_like(mask_new, dtype=bool)
    k = 0
    while np.any(mask_new != mask_old) & (k <= maxiters):

        mask_old = mask_new.copy()

        p = np.poly1d(np.polyfit(x[~mask_old], y[~mask_old], deg=deg, w=w[~mask_old]))

        y_fit = p(x)

        # Standardized residual
        residual = (y - y_fit) * w
        if use_relative:
            residual /= y_fit

        # Sigma clipping
        if maxiters == 0:
            threshold_lower, threshold_upper = None, None

        elif k < maxiters:
            residual[m] = np.nan
            residual_masked, threshold_lower, threshold_upper = sigma_clip(
                data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                maxiters=1, stdfunc='mad_std', axis=None, masked=True, 
                return_bounds=True, grow=grow)
            mask_new = residual_masked.mask

        k += 1

    return p, residual, threshold_lower, threshold_upper, mask_new


def Spline1D(x, y, w=None, m=None, order=3, n_piece=1, bbox=[None, None], maxiters=5, 
             sigma_lower=None, sigma_upper=None, grow=False, use_relative=False):
    """Fit 1-dimensional spline function.
    
    Knots are set equally spaced. Sigma clipping is used to mask bad data points.

    Parameters
    ----------
    x : array_like
        Input dimension of data points – must be increasing.

    y : array_like
        Input dimension of data points.

    w : array_like, optional
        Weights for spline fitting. Must be positive. If `None` (default), weights are 
        all equal.

    m : array_like, optional
        Initial mask for spline fitting. If `None` (default), data points are all 
        unmasked.

    order : int, optional
        Degree of the smoothing spline. Must be `5` >= ``order`` >= `1`.
        Default is `3`, a cubic spline.

    n_piece : int, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        Default is `1`.

    bbox : array_like, optional
        2-sequence specifying the boundary of the approximation interval. If `None` 
        (default), bbox = [x[0], x[-1]].

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

    use_relative : bool, optional
        If `True`, relative residual is used in sigma slipping instead of residual. 
        Default is `False`.

    Returns
    -------
    spl : callable
        Callable object returned by `~scipy.interpolate.LSQUnivariateSpline`.

    residual : `~numpy.ndarray`
        Residual or relative residual, depending on ``use_relative``.

    threshold_lower : scalar or `None`
        Lower threshold for clipping.

    threshold_upper : scalar or `None`
        Upper threshold for in clipping.

    master_mask : `~numpy.ndarray`
        Final mask after clipping.
    """

    if w is None:
        w = np.ones_like(y)

    if m is None:
        m = np.zeros_like(y, dtype=bool)

    _validateInteger(n_piece, 'n_piece', (1, None), (True, None))

    _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

    _validateBool(use_relative, 'use_relative')

    # Set ``n_piece`` equally spaced knots
    knots = x[~m][0] + np.arange(1, n_piece) * (x[~m][-1] - x[~m][0]) / n_piece

    # Spline fitting iteratively
    mask_new = m.copy()
    mask_old = np.ones_like(mask_new, dtype=bool)
    k = 0
    while np.any(mask_new != mask_old) & (k <= maxiters):

        mask_old = mask_new.copy()

        spl = interpolate.LSQUnivariateSpline(
            x=x[~mask_old], y=y[~mask_old], t=knots, w=w[~mask_old], bbox=bbox, 
            k=order, ext='extrapolate', check_finite=False)

        y_fit = spl(x)

        # Standardized residual
        residual = (y - y_fit) * w
        if use_relative:
            residual /= y_fit

        # Sigma clipping
        if maxiters == 0:
            threshold_lower, threshold_upper = None, None

        elif k < maxiters:
            residual[m] = np.nan
            residual_masked, threshold_lower, threshold_upper = sigma_clip(
                data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                maxiters=1, stdfunc='mad_std', axis=None, masked=True, 
                return_bounds=True, grow=grow)
            mask_new = residual_masked.mask

        k += 1

    return spl, residual, threshold_lower, threshold_upper, mask_new


def Spline2D(x, y, z, w=None, m=None, order=(3, 3), n_piece=(1, 1), 
             bbox=[None, None, None, None], maxiters=5, sigma_lower=None, 
             sigma_upper=None, axis=None, grow=False, use_relative=False):
    """Fit 2-dimensional spline function.
    
    Knots are set equally spaced. Sigma clipping is used to mask bad data points.

    Parameters
    ----------
    x : array_like
        Input dimension of data points – must be increasing.

    y : array_like
        Input dimension of data points.

    z : array_like
        Input dimension of data points.

    w : array_like or `None`, optional
        Weights for spline fitting. Must be positive. If `None` (default), weights are 
        all equal.

    m : array_like or `None`, optional
        Initial mask for spline fitting. If `None` (default), data points are all 
        unmasked.

    order : 2-tuple of int, optional
        Degree of the smoothing spline. Must be `5` >= ``order`` >= `1`.
        Default is (3, 3).

    n_piece : 2-tuple of int, optional
        Number of spline pieces. Lengths are all equal. Must be positive.
        Default is (1, 1).
        
    bbox : array_like, optional
        Sequence of length 4 specifying the boundary of the rectangular approximation 
        domain. By default, bbox=[x[0], x[-1], y[0], y[-1]].

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

    axis : `None` or int or 2-tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`, then the 
        flattened data will be used. ``axis`` is passed into the ``cenfunc`` and 
        ``stdfunc``. 
        Default is `None`.

    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits. (only applied along axis, if specified)

    use_relative : bool, optional
        If `True`, relative residual is used in sigma slipping instead of residual. 
        Default is `False`.

    Returns
    -------
    bispl : callable
        Callable object returned by `~scipy.interpolate.LSQBivariateSpline`.
    
    residual : `~numpy.ndarray`
        Residual or relative residual, depending on ``use_relative``.
    
    threshold_lower : scalar or array_like or None
        Lower threshold for clipping.
    
    threshold_upper : scalar or array_like or None
        Upper threshold for clipping.
    
    master_mask : `~numpy.ndarray`
        Final mask after clipping.
    """

    if w is None:
        w = np.ones_like(z)

    if m is None:
        m = np.zeros_like(z, dtype=bool)

    _validate1DArray(order, 'order', 2, True)

    _validate1DArray(n_piece, 'n_piece', 2, True)
    _validateInteger(n_piece[0], 'n_piece[0]', (1, None), (True, None))
    _validateInteger(n_piece[1], 'n_piece[1]', (1, None), (True, None))

    _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

    _validateBool(use_relative, 'use_relative')

    # Order
    xorder, yorder = order

    # Set equally spaced knots
    xpiece, ypiece = n_piece
    xknots = x[~m][0] + np.arange(1, xpiece) * (x[~m][-1] - x[~m][0]) / xpiece
    yknots = y[~m][0] + np.arange(1, ypiece) * (y[~m][-1] - y[~m][0]) / ypiece
    
    # 2-dimensional spline fitting iteratively
    mask_new = m.copy()
    mask_old = np.ones_like(mask_new, dtype=bool)
    k = 0
    while np.any(mask_new != mask_old) & (k <= maxiters):

        mask_old = mask_new.copy()

        bispl = interpolate.LSQBivariateSpline(
            x=x[~mask_old], y=y[~mask_old], z=z[~mask_old], tx=xknots, ty=yknots, 
            w=w[~mask_old], bbox=bbox, kx=xorder, ky=yorder, eps=None)

        z_fit = bispl(x.flatten(), y.flatten(), grid=False).reshape(x.shape)

        # Standardized residual
        residual = (z - z_fit) * w
        if use_relative:
            residual /= z_fit

        # Sigma clipping
        if maxiters == 0:
            threshold_lower, threshold_upper = None, None

        elif k < maxiters:
            residual[m] = np.nan
            residual_masked, threshold_lower, threshold_upper = sigma_clip(
                data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
                maxiters=1, stdfunc='mad_std', axis=axis, masked=True, 
                return_bounds=True, grow=grow)
            mask_new = residual_masked.mask

        k += 1

    return bispl, residual, threshold_lower, threshold_upper, mask_new


def GaussianSmoothing2D(x, y, z, sigma, m=None, maxiters=5, sigma_lower=None, 
                        sigma_upper=None, axis=None, grow=False, use_relative=False):
    """2-dimensional Gaussian smoothing.

    Parameters
    ----------
    x : array_like
        Input dimension of data points.

    y : array_like
        Input dimension of data points.

    z : array_like
        Input dimension of data points.

    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard deviations of the Gaussian 
        filter are given for each axis as a sequence, or as a single number, in which 
        case it is equal for both axes.

    m : array_like, optional
        Initial mask for Gaussian smoothing. If `None` (default), data points are all 
        unmasked.

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

    axis : `None` or int or 2-tuple of int, optional
        The axis or axes along which to sigma clip the data. If `None`, then the 
        flattened data will be used. ``axis`` is passed into the ``cenfunc`` and 
        ``stdfunc``. 
        Default is `None`.

    grow : scalar or `False`, optional
        Radius within which to mask the neighbouring pixels of those that fall outwith 
        the clipping limits. (only applied along axis, if specified)

    use_relative : bool, optional
        If `True`, relative residual is used in sigma slipping instead of residual. 
        Default is `False`.

    Returns
    -------
    z_smoothed : `~numpy.ndarray`
        Smoothed data.

    residual : `~numpy.ndarray`
        Residual or relative residual, depending on ``use_relative``.

    threshold_lower : scalar or array_like or None
        Lower threshold for clipping.
    
    threshold_upper : scalar or array_like or None
        Upper threshold for clipping.

    master_mask : `~numpy.ndarray`
        Final mask after clipping.
    """

    if m is None:
        m = np.zeros(z.shape, dtype=bool)

    _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

    _validateBool(use_relative, 'use_relative')

    # Convolve with Gaussian kernel iteratively
    mask_new = m.copy()
    mask_old = np.ones_like(mask_new, dtype=bool)
    k = 0
    while np.any(mask_new != mask_old) & (k <= maxiters):

        mask_old = mask_new.copy()

        # Interpolate
        interpolated_z = interpolate.griddata(
            points=(x[~mask_old], y[~mask_old]), values=z[~mask_old],
            xi=(x.flatten(), y.flatten()), method='nearest').reshape(x.shape)

        # Smooth
        z_smoothed = gaussian_filter(
            input=interpolated_z, sigma=sigma, order=0, mode='reflect')

        # Residual
        residual = z - z_smoothed
        if use_relative:
            residual /= z_smoothed

        # Sigma clipping
        if maxiters == 0:
            threshold_lower, threshold_upper = None, None

        elif k < maxiters:
            residual[m] = np.nan
            residual_masked, threshold_lower, threshold_upper = sigma_clip(
                data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
                maxiters=1, stdfunc='mad_std', axis=axis, masked=True, 
                return_bounds=True, grow=grow)
            mask_new = residual_masked.mask

        k += 1

    return z_smoothed, residual, threshold_lower, threshold_upper, mask_new


# def _SavgolFilter1D(y, window_length, polyorder, deriv=0, delta=1.0, axis=-1, 
#                    mode='interp', cval=0.0):
#     """2-dimensional Gaussian smoothing.

#     Parameters
#     ----------
#     x : array_like
#         Input dimension of data points.

#     y : array_like
#         Input dimension of data points.

#     z : array_like
#         Input dimension of data points.

#     sigma : scalar or sequence of scalars
#         Standard deviation for Gaussian kernel. The standard deviations of the Gaussian 
#         filter are given for each axis as a sequence, or as a single number, in which 
#         case it is equal for both axes.

#     m : array_like, optional
#         Initial mask for Gaussian smoothing. If `None` (default), data points are all 
#         unmasked.

#     maxiters : int, optional
#         Maximum number of sigma-clipping iterations to perform. If convergence is 
#         achieved prior to ``maxiters`` iterations, the clipping iterations will stop. 
#         Must be >= `0`. 
#         Default is `5`.

#     sigma_lower : scalar or `None`, optional
#         Number of standard deviations to use as the lower bound for the clipping limit. 
#         If `None` (default), `3` is used.

#     sigma_upper : scalar or `None`, optional
#         Number of standard deviations to use as the upper bound for the clipping limit. 
#         If `None` (default), `3` is used.

#     axis : `None` or int or 2-tuple of int, optional
#         The axis or axes along which to sigma clip the data. If `None`, then the 
#         flattened data will be used. ``axis`` is passed into the ``cenfunc`` and 
#         ``stdfunc``. 
#         Default is `None`.

#     grow : scalar or `False`, optional
#         Radius within which to mask the neighbouring pixels of those that fall outwith 
#         the clipping limits. (only applied along axis, if specified)

#     use_relative : bool, optional
#         If `True`, relative residual is used in sigma slipping instead of residual. 
#         Default is `False`.

#     Returns
#     -------
#     z_smoothed : `~numpy.ndarray`
#         Smoothed data.

#     residual : `~numpy.ndarray`
#         Residual or relative residual, depending on ``use_relative``.

#     threshold_lower : scalar or array_like or None
#         Lower threshold for clipping.
    
#     threshold_upper : scalar or array_like or None
#         Upper threshold for clipping.

#     master_mask : `~numpy.ndarray`
#         Final mask after clipping.
#     """

#     if m is None:
#         m = np.zeros(z.shape, dtype=bool)

#     _validateInteger(maxiters, 'maxiters', (0, None), (True, None))

#     _validateBool(use_relative, 'use_relative')

#     # Convolve with Gaussian kernel iteratively
#     mask_new = m.copy()
#     mask_old = np.ones_like(mask_new, dtype=bool)
#     k = 0
#     while np.any(mask_new != mask_old) & (k <= maxiters):

#         mask_old = mask_new.copy()

#         # Interpolate
#         interpolated_z = interpolate.griddata(
#             points=(x[~mask_old], y[~mask_old]), values=z[~mask_old],
#             xi=(x.flatten(), y.flatten()), method='nearest').reshape(x.shape)

#         # Smooth
#         z_smoothed = gaussian_filter(
#             input=interpolated_z, sigma=sigma, order=0, mode='reflect')

#         # Residual
#         residual = z - z_smoothed
#         if use_relative:
#             residual /= z_smoothed

#         # Sigma clipping
#         if maxiters == 0:
#             threshold_lower, threshold_upper = None, None

#         elif k < maxiters:
#             residual[m] = np.nan
#             residual_masked, threshold_lower, threshold_upper = sigma_clip(
#                 data=residual, sigma_lower=sigma_lower, sigma_upper=sigma_upper,
#                 maxiters=1, stdfunc='mad_std', axis=axis, masked=True, 
#                 return_bounds=True, grow=grow)
#             mask_new = residual_masked.mask

#         k += 1

#     return z_smoothed, residual, threshold_lower, threshold_upper, mask_new
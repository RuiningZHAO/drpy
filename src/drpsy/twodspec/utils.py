import warnings

# NumPy
import numpy as np
# Scipy
from scipy import interpolate
# drpsy
from drpsy.validate import _validateNDArray

__all__ = ['invertCoordinateMap']

# todo: deal with griddata
def invertCoordinateMap(U=None, V=None):
    """Invert coordinate map.
    
    U(X, Y) and V(X, Y) -> X(U, V) and Y(U, V).
    
    Parameters
    ----------
    U, V : `~numpy.ndarray` or `None`, optional
        Coordinate maps to be inverted.

    Returns
    -------
    X, Y : `~numpy.ndarray`
        Inverted coordinate maps.
    """

    if (U is None) & (V is None):

        X, Y = None, None

    elif U is None:

        _validateNDArray(V, 'V', 2)

        warnings.warn(
            'A one-to-one mapping is assumed for ``U``, i.e., U = X.', RuntimeWarning)

        n_row, n_col = V.shape

        idx_row, idx_col = np.arange(n_row), np.arange(n_col)

        X = np.tile(np.arange(n_col), (n_row, 1))

        Y = np.zeros((n_row, n_col))
        for i in idx_col:
            Y[:, i] = interpolate.interp1d(
                x=V[:, i], y=idx_row, bounds_error=False, fill_value='extrapolate', 
                assume_sorted=True)(idx_row)

    elif V is None:

        _validateNDArray(U, 'U', 2)

        warnings.warn(
            'A one-to-one mapping is assumed for ``V``, i.e., V = Y.', RuntimeWarning)

        n_row, n_col = U.shape

        idx_row, idx_col = np.arange(n_row), np.arange(n_col)

        Y = np.tile(np.arange(n_row), (n_col, 1)).T

        X = np.zeros((n_row, n_col))
        for i in idx_row:
            X[i] = interpolate.interp1d(
                x=U[i], y=idx_col, bounds_error=False, fill_value='extrapolate', 
                assume_sorted=True)(idx_col)

    else: # !!! DO NOT USE THIS !!!

        _validateNDArray(U, 'U', 2)
        _validateNDArray(V, 'V', 2)

        if U.shape != V.shape:
            raise ValueError('``U`` and ``V`` should have the same shape.')

        n_row, n_col = U.shape

        X = np.tile(np.arange(n_col), (n_row, 1))
        Y = np.tile(np.arange(n_row), (n_col, 1)).T

        nX = interpolate.griddata(
            (U.flatten(), V.flatten()), X.flatten(), (X, Y), method='linear', 
            fill_value=np.nan, rescale=False)

        nY = interpolate.griddata(
            (U.flatten(), V.flatten()), Y.flatten(), (X, Y), method='linear', 
            fill_value=np.nan, rescale=False)
        
        X, Y = nX, nY
    
    return X, Y
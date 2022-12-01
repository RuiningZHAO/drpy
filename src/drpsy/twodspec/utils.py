# NumPy
import numpy as np
# Scipy
from scipy import interpolate
# drpsy
from drpsy.validate import _validateString, _validateNDArray

__all__ = ['invertCoordinateMap']


def invertCoordinateMap(slit_along, coordinate):
    """Invert coordinate map.

    U(X, Y) and V(X, Y) -> X(U, V) and Y(U, V).

    Parameters
    ----------
    slit_along : str
        `col` or `row`, and 
        - if `col`, ``V`` = ``Y`` is assumed.
        - if `row`, ``U`` = ``X`` is assumed.

    coordinate : `~numpy.ndarray`
        One of the coordinate maps to be inverted (i.e., ``U`` or ``V``). A one-to-one 
        mapping is assumed for the other.

    Returns
    -------
    X, Y : `~numpy.ndarray`
        Inverted coordinate maps.
    """

    _validateString(slit_along, 'slit_along', ['col', 'row'])

    if slit_along == 'col':

        U = _validateNDArray(coordinate, 'coordinate', 2)

        n_row, n_col = U.shape

        idx_row, idx_col = np.arange(n_row), np.arange(n_col)

        Y = np.tile(np.arange(n_row), (n_col, 1)).T

        X = np.zeros((n_row, n_col))
        for i in idx_row:
            X[i] = interpolate.interp1d(
                x=U[i], y=idx_col, bounds_error=False, fill_value='extrapolate', 
                assume_sorted=True)(idx_col)

    else:

        V = _validateNDArray(coordinate, 'coordinate', 2)

        n_row, n_col = V.shape

        idx_row, idx_col = np.arange(n_row), np.arange(n_col)

        X = np.tile(np.arange(n_col), (n_row, 1))

        Y = np.zeros((n_row, n_col))
        for i in idx_col:
            Y[:, i] = interpolate.interp1d(
                x=V[:, i], y=idx_row, bounds_error=False, fill_value='extrapolate', 
                assume_sorted=True)(idx_row)
    
    return X, Y
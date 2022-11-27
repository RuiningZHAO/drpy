import warnings

# NumPy
import numpy as np
# Scipy
from scipy import interpolate
# AstroPy
from astropy.table import Table
from astropy.nddata import CCDData

from .validate import (_validateBool, _validateNDArray, _validateCCDData, 
                       _validateCCDList)

__all__ = ['imstatistics', 'invertCoordinateMap']


def imstatistics(ccdlist, verbose=False):
    """Image statistics.
    
    Parameters
    ----------
    ccdlist : Iterable or `~astropy.nddata.CCDData` or `~numpy.ndarray`
        A 2-dimensional image or an iterable object consist of 2-dimensional 
        images, the statistics of which are calculated.

    verbose : bool, optional
        If `True`, statistics table is printted before return.
        Default is `False`.
    
    Returns
    -------
    stats : `~astropy.table.table.Table`
        Statistics table.
    """

    if not isinstance(ccdlist, (np.ndarray, CCDData)):
        nccdlist = _validateCCDList(ccdlist, 1, 2) # dimension is set to 2
    else:
        nccdlist = [_validateCCDData(ccdlist, 'ccd')]
    
    n_image = len(nccdlist)

    # Construct table
    rows = list()
    for i in range(n_image):
        
        ccd = nccdlist[i]
        
        if 'FILENAME' in ccd.header:
            file_name = ccd.header['FILENAME']
        else:
            file_name = f'image {i + 1}'

        # !!! mask is effective here !!!
        rows.append(
            [file_name, ccd.shape, np.mean(ccd.data), np.std(ccd.data, ddof=1), 
             np.min(ccd.data), np.max(ccd.data)])

    stats = Table(rows=rows, names=['IMAGE', 'SHAPE', 'MEAN', 'STDDEV', 'MIN', 'MAX'])
    
    # Format
    for colname in ['MEAN', 'STDDEV', 'MIN', 'MAX']:
        stats[colname].format = "g"

    _validateBool(verbose, 'verbose')

    if verbose: stats.pprint_all()

    return stats

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
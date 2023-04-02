# NumPy
import numpy as np
# Scipy
from scipy import interpolate
# AstroPy
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.modeling.models import Gaussian2D
# photutils
from photutils.aperture import (CircularAperture, RectangularAperture, 
                                aperture_photometry)
# drpy
from drpy.validate import _validateString, _validateNDArray

__all__ = ['invertCoordinateMap', 'getSlitLoss']


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


def getSlitLoss(fwhm, slit_width):
    """Calculate slit loss.
    
    Parameters
    ----------
    fwhm: scalar
        Seeing.
    
    slit_width: scalar
        Slit width.
    
    Returns
    -------
    slit_loss: scalar
        Slit loss.
    """
    
    sigma = fwhm * gaussian_fwhm_to_sigma
    amplitude = (1 / (2 * np.pi * sigma**2))
    
    gaussian2d = Gaussian2D(
        amplitude=amplitude, x_mean=0, y_mean=0, x_stddev=sigma, y_stddev=sigma, 
        theta=0)
    
    ny = nx = np.int64(10 * sigma)
    Y, X = np.mgrid[-ny:ny+1, -nx:nx+1]
    data = gaussian2d(X, Y)
    
    (y, x) = np.unravel_index(np.argmax(data, axis=None), data.shape)
    
    # Aperture photometry
    rect_aper = RectangularAperture((x, y), w=slit_width, h=data.shape[0])
    flux_slit = aperture_photometry(data, rect_aper, method='exact')['aperture_sum'][0]
    
    # circ_aper = CircularAperture((x, y), r=data.shape[0] / 2)
    # flux_total = aperture_photometry(data, circ_aper, method='exact')['aperture_sum'][0]
    
    slit_loss = 1 - flux_slit
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.imshow(data, )
    # rect_aper.plot(ax=ax, color='red', lw=2)
    # circ_aper.plot(ax=ax, color='red', lw=2)
    # plt.show()
    
    return slit_loss
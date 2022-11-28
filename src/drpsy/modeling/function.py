# Numpy
import numpy as np

__all__ = ['Gaussian1D', 'CircularGaussian']


def Gaussian1D(x, a, x0, sigma):
    """1D Gaussian function.

    Parameters
    ----------
    x : scalar or `~numpy.ndarray`
        Value or values to evaluate the Gaussian at.

    a : scalar
        Amplitude.

    x0 : scalar
        Gaussian center.

    sigma : scalar
        Standard deviation.

    Returns
    -------
    out: scalar or `~numpy.ndarray`
        The Gaussian function evaluated at provided x.
    """

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def CircularGaussian(xy, a, x0, y0, sigma, b):
    """Circular Gaussian function.

    Parameters
    ----------
    xy : scalar or `~numpy.ndarray`
        Value or values to evaluate the Gaussian at.

    a : scalar
        Amplitude.

    x0, y0 : scalar
        Gaussian center.

    sigma : scalar
        Standard deviation.

    b : scalar
        Offset.

    Returns
    -------
    out: scalar or `~numpy.ndarray`
        The circular Gaussian function evaluated at provided (x, y).
    """

    x, y = xy

    return a * np.exp(-(x - x0)**2 / (2 * sigma**2) - (y - y0)**2 / (2 * sigma**2)) + b
"""
drspy.modeling: fitting methods.
"""

# AstroPy
from astropy import config as _config

from .core import Poly1D, Spline1D, Spline2D, GaussianSmoothing2D

__all__ = ['Poly1D', 'Spline1D', 'Spline2D', 'GaussianSmoothing2D', 'conf']


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `drspy.modeling`."""

    pass


conf = Conf()
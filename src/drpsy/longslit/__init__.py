"""
drpsy.longslit: a toolkit to reduce longslit spectroscopy data.
"""

# AstroPy
from astropy import config as _config

from .core import (response, illumination, align, fitcoords, dispcor, trace, 
                   background, extract, sensfunc, calibrate1d, calibrate2d)

__all__ = ['response', 'illumination', 'align', 'fitcoords', 'dispcor', 'trace', 
           'background', 'extract', 'sensfunc', 'calibrate1d', 'calibrate2d', 'conf']


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `drpsy.longslit`."""

    pass


conf = Conf()
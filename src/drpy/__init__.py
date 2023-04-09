"""
drpy: a data reduction toolkit for astronomical photometry and spectroscopy
"""

import os

# AstroPy
from astropy import config as _config

__all__ = ['__version__', 'conf']

__version__ = '0.0.1.6'


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `drpy`."""

    # Unit
    unit_ccddata = _config.ConfigItem(
        'adu', 'Default unit for `~astropy.nddata.CCDData` objects.')

    unit_spectral_axis = _config.ConfigItem(
        'pixel', 'Default unit for `~specutils.Spectrum1D` objects.')

    unit_flux = _config.ConfigItem(
        'count', 'Default unit for `~specutils.Spectrum1D` objects.')

    dtype_ccddata = _config.ConfigItem(
        'float32', 'Default dtype for `~astropy.nddata.CCDData` objects.')

    # Plot
    show = _config.ConfigItem(False, 'Whether to show plots.')

    save = _config.ConfigItem(False, 'Whether to save plots.')

    path = _config.ConfigItem(os.getcwd(), 'Where to save plots.')


conf = Conf()

del _config
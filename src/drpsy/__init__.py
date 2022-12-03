"""
drpsy: a data reduction toolkit for astronomical photometry and spectroscopy
"""

import os

from astropy import config as _config

__all__ = ['__version__', 'conf']

__version__ = '0.0.4.4'


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `drpsy`."""

    # Unit
    unit_ccddata = _config.ConfigItem(
        'adu', 'Default unit for `~astropy.nddata.CCDData` objects.', cfgtype='string')

    unit_spectral_axis = _config.ConfigItem(
        'pixel', 'Default unit for `~specutils.Spectrum1D` objects.', cfgtype='string')

    unit_flux = _config.ConfigItem(
        'count', 'Default unit for `~specutils.Spectrum1D` objects.', cfgtype='string')

    # Plot
    show = _config.ConfigItem(False, 'Whether to show plots.', cfgtype='boolean')

    save = _config.ConfigItem(False, 'Whether to save plots.', cfgtype='boolean')

    path = _config.ConfigItem(os.getcwd(), 'Where to save plots.', cfgtype='string')


conf = Conf()

del _config
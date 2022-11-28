"""
drpsy: a pythonic Data Reduction toolbox for Spectroscopy and Photometry
"""

# AstroPy
from astropy import config as _config

from .core import CCDDataList, concatenate, transform


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

conf = Conf()

__version__ = '0.0.3.1'

__all__ = ['CCDDataList', 'concatenate', 'transform', 'conf']
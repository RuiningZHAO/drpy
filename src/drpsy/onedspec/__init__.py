"""
drpsy.onedspec: a toolkit to reduce 1-dimensional spectroscopy data.
"""

from .core import dispcor, sensfunc, calibrate1d
from .io import (saveSpectrum1D, loadSpectrum1D, loadStandardSpectrum, 
                 loadExtinctionCurve)

__all__ = ['dispcor', 'sensfunc', 'calibrate1d', 'saveSpectrum1D', 'loadSpectrum1D', 
           'loadStandardSpectrum', 'loadExtinctionCurve']
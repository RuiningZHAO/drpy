"""
drpsy.onedspec: a toolkit to reduce 1-dimensional spectroscopy data.
"""

from .core import dispcor, sensfunc, calibrate1d

__all__ = ['dispcor', 'sensfunc', 'calibrate1d']
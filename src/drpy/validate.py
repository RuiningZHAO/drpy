import operator, os, warnings
from collections.abc import Iterable
from copy import deepcopy

# NumPy
import numpy as np
# AstroPy
import astropy.units as u
from astropy.nddata import CCDData, StdDevUncertainty
# specutils
from specutils import Spectrum1D
# drpy
from drpy import conf

# Configurations
unit_ccddata = u.Unit(conf.unit_ccddata)
unit_spectral_axis = u.Unit(conf.unit_spectral_axis)
unit_flux = u.Unit(conf.unit_flux)

__all__ = ['_validateBool', '_validateString', '_validateRange', '_validateInteger', 
           '_validateLength', '_validate1DArray', '_validateNDArray', 
           '_validateCCDData', '_validateSpectrum1D', '_validateCCDList', 
           '_validateCCD', '_validateSpectrum', '_validateBins', '_validateAperture', 
           '_validatePath']
# Types

def _validateBool(arg, name):
    """Validate a `bool` object."""

    if not isinstance(arg, (bool, np.bool_)):
        raise TypeError(
            'Invalid type `{}` for ``{}``. `bool` is expected.'
            .format(type(arg), name))

    return True


def _validateString(arg, name, value_list=None):
    """Validate a `str` object."""

    if not isinstance(arg, (str, np.str_)):
        raise TypeError(
            'Invalid type `{}` for ``{}``. `str` is expected.'.format(type(arg), name))

    if value_list is not None:
        if arg not in value_list:
            raise ValueError('``{}`` should be in `{}`.'.format(name, value_list))

    return True


# With properties

def _validateRange(arg, name, value_range=(None, None), is_closed=(None, None)):
    """Validate whether ``arg`` is in the specified range."""

    if (value_range[0] is not None) & (value_range[1] is not None):
        if is_closed[0] & is_closed[1]:
            if (arg < value_range[0]) | (value_range[1] < arg):
                raise ValueError(
                    '{} must be in range [{}, {}]. {} is given.'.format(
                        name, value_range[0], value_range[1], arg)
                )
        elif is_closed[0]:
            if (arg < value_range[0]) | (value_range[1] <= arg):
                raise ValueError(
                    '{} must be in range [{}, {}). {} is given.'.format(
                        name, value_range[0], value_range[1], arg)
                )
        elif is_closed[1]:
            if (arg <= value_range[0]) | (value_range[1] < arg):
                raise ValueError(
                    '{} must be in range ({}, {}]. {} is given.'.format(
                        name, value_range[0], value_range[1], arg)
                )
        else:
            if (arg <= value_range[0]) | (value_range[1] <= arg):
                raise ValueError(
                    '{} must be in range ({}, {}). {} is given.'.format(
                        name, value_range[0], value_range[1], arg)
                )
    elif value_range[0] is not None:
        if is_closed[0]:
            if arg < value_range[0]:
                raise ValueError(
                    '{} must be in range [{}, inf). {} is given.'.format(
                        name, value_range[0], arg)
                )
        else:
            if arg <= value_range[0]:
                raise ValueError(
                    '{} must be in range ({}, inf). {} is given.'.format(
                        name, value_range[0], arg)
                )
    elif value_range[1] is not None:
        if is_closed[1]:
            if arg > value_range[1]:
                raise ValueError(
                    '{} must be in range (-inf, {}]. {} is given.'.format(
                        name, value_range[1], arg)
                )
        else:
            if arg >= value_range[1]:
                raise ValueError(
                    '{} must be in range (-inf, {}). {} is given.'.format(
                        name, value_range[1], arg)
                )

    return True


def _validateInteger(arg, name, value_range=(None, None), is_closed=(None, None)):
    """Validate an `int` object."""

    try:
        arg = operator.index(arg)

    except TypeError:
        raise TypeError(
            'Invalid type `{}` for ``{}``. `int` is expected.'.format(type(arg), name))

    _validateRange(arg, name, value_range, is_closed)

    return True


def _validateLength(arg, name, length):
    """Validate length of an array."""

    narg = np.array(arg)

    if narg.shape[0] < length:
        raise ValueError(
            '``{}`` should be of length {}. {} is given.'.format(
            name, length, narg.shape[0]))

    elif narg.shape[0] > length:
        narg = narg[:length]
    
    return narg


def _validate1DArray(arg, name, length, accept_scalar):
    """Validate a 1-dimensional array."""

    if np.ndim(arg) == 0:
        if accept_scalar:
            narg = np.zeros(length) + arg

        else:
            raise ValueError('``{}`` must be 1-dimensional.'.format(name))

    elif np.ndim(arg) == 1:
        narg = _validateLength(arg, name, length)
    
    else:
        raise ValueError('``{}`` must be 1-dimensional, when an array.'.format(name))
    
    return narg


def _validateNDArray(arg, name, ndim):
    """Validate a 2-dimensional array."""

    if np.ndim(arg) == ndim:
        narg = np.array(arg)

    else:
        raise ValueError(
            '``{}`` must be {}-dimensional.'.format(name, ndim))

    return narg


# Specific arguments

def _validateCCDData(ccd, name):
    """Validate a CCD image."""

    if not isinstance(ccd, (np.ndarray, CCDData)):
        raise TypeError('Invalid type `{}` for ``{}``.'.format(type(ccd), name))
    # May have (or not have) uncertainty frame or mask frame.
    elif isinstance(ccd, CCDData):
        nccd = ccd.copy()
    # Do not have uncertainty frame. May have (or not have) mask frame.
    else:
        nccd = CCDData(ccd.copy(), unit=unit_ccddata)

    return nccd


def _validateSpectrum1D(spectrum, name):
    """Validate a spectrum."""

    if not isinstance(spectrum, (np.ndarray, Spectrum1D)):
        raise TypeError('Invalid type `{}` for ``{}``.'.format(type(spectrum), name))
    
    elif isinstance(spectrum, Spectrum1D):
        new_spectrum = deepcopy(spectrum)
    
    else:

        if spectrum.ndim == 1:

            if np.ma.isMaskedArray(spectrum):
                flux = spectrum.data * unit_flux
                mask = spectrum.mask

            else:
                flux = spectrum * unit_flux
                mask = None

            new_spectrum = Spectrum1D(flux=flux, mask=mask)

        elif spectrum.ndim == 2:

            n_spec = spectrum.shape[0]

            if np.ma.isMaskedArray(spectrum):

                mask = spectrum.mask[:3].any(axis=0)

                # flux
                if n_spec == 1:
                    spectral_axis = np.arange(spectrum.shape[1]) * unit_spectral_axis
                    flux = spectrum.data[0] * unit_flux
                    uncertainty = None

                # spectral axis, flux
                elif n_spec == 2:
                    spectral_axis = spectrum.data[0] * unit_spectral_axis
                    flux = spectrum.data[1] * unit_flux
                    uncertainty = None

                # spectral axis, flux, uncertainty
                else:
                    spectral_axis = spectrum.data[0] * unit_spectral_axis
                    flux = spectrum.data[1] * unit_flux
                    uncertainty = StdDevUncertainty(spectrum.data[2])

            else:

                mask = None

                # flux
                if n_spec == 1:
                    spectral_axis = np.arange(spectrum.shape[1]) * unit_spectral_axis
                    flux = spectrum[0] * unit_flux
                    uncertainty = None

                # spectral axis, flux
                elif n_spec == 2:
                    spectral_axis = spectrum[0] * unit_spectral_axis
                    flux = spectrum[1] * unit_flux
                    uncertainty = None

                # spectral axis, flux, uncertainty
                else:
                    spectral_axis = spectrum[0] * unit_spectral_axis
                    flux = spectrum[1] * unit_flux
                    uncertainty = StdDevUncertainty(spectrum[2])

            new_spectrum = Spectrum1D(
                spectral_axis=spectral_axis, flux=flux, uncertainty=uncertainty, 
                mask=mask)

        else:
            raise ValueError(
                '``{}`` must be 1 or 2-dimensional, when an array.'.format(name))
    
    return new_spectrum


def _validateCCDList(ccdlist, n_frame, n_dim):
    """Validate a list of frames."""
    
    # Type
    if isinstance(ccdlist, Iterable):

        # Length
        if len(ccdlist) < n_frame:
            raise ValueError(
                f'``ccdlist`` should contain at least {n_frame} frames.')

        # Type and dimension of elements
        nccdlist = list()
        for ccd in ccdlist:
            nccd = _validateCCDData(ccd, 'ccd')
            if nccd.ndim != n_dim:
                raise ValueError(
                    f'A dimension of `{n_dim}` is expected. `{nccd.ndim}` is given.')
            else:
                nccdlist.append(nccd)

    else:
        raise TypeError(f'Invalid type `{type(ccdlist)}` for ``ccdlist``.')
    
    return nccdlist


def _validateCCD(ccd, name, use_uncertainty, use_mask, transpose, asWeight=False):
    """Validate a CCD image."""
    
    nccd = _validateCCDData(ccd, name)

    # Data
    data_arr = nccd.data.copy()
    
    # Uncertainty
    if use_uncertainty:
        if nccd.uncertainty is None:
            warnings.warn(
                'The input uncertainty is unavailable. All set to zero.', 
                RuntimeWarning)
            uncertainty_arr = (
                np.zeros_like(data_arr) if not asWeight else np.ones_like(data_arr))
        else:
            uncertainty_arr = nccd.uncertainty.array.copy()
    else:
        uncertainty_arr = (
            np.zeros_like(data_arr) if not asWeight else np.ones_like(data_arr))
    
    # Mask
    if use_mask:
        if nccd.mask is None:
            warnings.warn(
                'The input mask is unavailable. All set unmasked.', RuntimeWarning)
            mask_arr = np.zeros_like(data_arr, dtype=bool)
        else:
            mask_arr = nccd.mask.copy()
    else:
        mask_arr = np.zeros_like(data_arr, dtype=bool)

    if transpose:
        data_arr = data_arr.T
        uncertainty_arr = uncertainty_arr.T
        mask_arr = mask_arr.T
    
    return nccd, data_arr, uncertainty_arr, mask_arr


def _validateSpectrum(spectrum, name, use_uncertainty, use_mask):
    """Validate a spectrum."""

    new_spectrum = _validateSpectrum1D(spectrum, name)
    
    # Data
    spectral_axis = new_spectrum.spectral_axis.value.copy()
    flux = new_spectrum.flux.value.copy()
    
    # Uncertainty
    if use_uncertainty:
        if new_spectrum.uncertainty is None:
            warnings.warn(
                'The input uncertainty is unavailable. All set to zero.', 
                RuntimeWarning)
            uncertainty = np.zeros_like(flux)
        else:
            uncertainty = new_spectrum.uncertainty.array.copy()
    else:
        uncertainty = np.zeros_like(flux)
    
    # Mask
    if use_mask:
        if new_spectrum.mask is None:
            warnings.warn(
                'The input mask is unavailable. All set unmasked.', RuntimeWarning)
            mask = np.zeros_like(flux, dtype=bool)
        else:
            mask = new_spectrum.mask.copy()
    else:
        mask = np.zeros_like(flux, dtype=bool)
    
    return new_spectrum, spectral_axis, flux, uncertainty, mask


def _validateBins(bins, length, isWidth=False):
    """Validate bins and derive bin edges."""

    if np.ndim(bins) == 0:
        # Validate
        _validateInteger(bins, 'bins', (1, length), (True, True))
        # Generate bin edges
        if isWidth:
            bin_edges = np.hstack([np.arange(0, length, bins), length])
        else:
            bin_edges = np.linspace(0, length, bins + 1).astype(int)

    elif np.ndim(bins) == 1:
        # Generate bin edges
        bin_edges = np.array(bins).astype(int)
        # Validate
        _validateInteger(bin_edges.min(), 'bin_edges', (0, length), (True, True))
        _validateInteger(bin_edges.max(), 'bin_edges', (0, length), (True, True))
        if np.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError('``bins`` must increase monotonically, when an array')

    else:
        raise ValueError('``bins`` must be 1-dimensional, when an array')

    bin_edges = np.vstack([bin_edges[:-1], bin_edges[1:]]).T
    n_bin = bin_edges.shape[0]
    loc_bin = (bin_edges.sum(axis=1) - 1) / 2

    return bin_edges, loc_bin, n_bin


def _validateAperture(aperture, name):
    """Validate aperture width."""

    if np.ndim(aperture) == 0:

        _validateRange(aperture, name, (0, None), (False, None))

        aperture = _validate1DArray(aperture, name, 2, True) / 2

    elif np.ndim(aperture) == 1:

        aperture = _validate1DArray(aperture, name, 2, True)

        _validateRange(
            aperture.sum(), f'{name}.sum()', (0, None), (False, None))

    else:
        raise ValueError(f'``{name}`` must be 1-dimensional, when an array')

    return aperture

# todo: slugify
def _validatePath(path, title, extension='.png'):
    """Validate path arguments."""

    if path is None:
        path = os.getcwd()

    elif not os.path.exists(path):
        raise OSError(f'Path {path} does not exist.')

    if isinstance(title, str):
        fig_name = f'{title}{extension}'.replace(' ', '_')
        fig_path = os.path.join(path, fig_name)

    else:
        fig_path = list()
        for t in title:
            fig_name = f'{t}{extension}'.replace(' ', '_')
            fig_path.append(os.path.join(path, fig_name))

    return fig_path
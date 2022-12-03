import os
from glob import glob

# NumPy
import numpy as np
# AstroPy
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from astropy.nddata import StdDevUncertainty
# specutils
from specutils import Spectrum1D
# drpsy
from drpsy.validate import _validateSpectrum1D

__all__ = ['saveSpectrum1D', 'loadSpectrum1D', 'loadStandardSpectrum', 
           'loadExtinctionCurve']


def _Spectrum1D_to_hdu(spectrum1d, header):
    """Convert a `~specutils.Spectrum1D` object to hdu."""

    # spectral_axis
    unit = spectrum1d.spectral_axis.unit.to_string('fits')
    array = spectrum1d.spectral_axis.value
    spectral_axis = fits.Column(
        name='spectral_axis', format='D', unit=unit, array=array)
    
    # flux
    unit = spectrum1d.flux.unit.to_string('fits')
    array = spectrum1d.flux.value

    if array.ndim <= 1:
        format = 'D'
        dim = None

    else:
        format = f'{np.prod(array.shape[:-1])}D'
        dim = '('
        for i in array.shape[:-1]:
            dim += f'{i},'
        dim = dim[:-1] + ')'

    flux = fits.Column(
        name='flux', format=format, unit=unit, dim=dim, array=array.T)

    columns = [spectral_axis, flux]

    # uncertainty
    if spectrum1d.uncertainty is not None:
        array = spectrum1d.uncertainty.array
        uncertainty = fits.Column(
            name='uncertainty', format=format, unit=unit, dim=dim, array=array.T)
        columns.append(uncertainty)

    # mask
    if spectrum1d.mask is not None:
        array = spectrum1d.mask
        mask = fits.Column(
            name='mask', format=format, dim=dim, array=array.T)
        columns.append(mask)

    hdu = fits.BinTableHDU.from_columns(columns, header)

    hdu.header['EXTNAME'] = 'spec'
    hdu.header['MODIFIED'] = (
        '{}'.format(Time.now().to_value('iso', subfmt='date_hm')), 'last modified')

    return hdu


def saveSpectrum1D(file_name, spectrum1d):
    """Save a `~specutils.Spectrum1D` object to file.
    
    Parameters
    ----------
    file_name : str
        File name.
    
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        The spectrum to be saved.
    """

    new_spectrum1d = _validateSpectrum1D(spectrum1d, 'spectrum1d')
    
    # header
    if 'header' in new_spectrum1d.meta:
        header = new_spectrum1d.meta['header']
    else:
        header = None

    hdu = _Spectrum1D_to_hdu(new_spectrum1d, header)

    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])

    hdulist.writeto(file_name, overwrite=True)

    return None

# todo: Table -> QTable ???
def loadSpectrum1D(file_name, ext=1, **kwargs):
    """Load tabular spectrum to `~specutils.Spectrum1D`.
    
    Parameters
    ----------
    file_name : str
        File name.

    ext : int or str, optional
        Extension.
    
    Returns
    -------
    spectrum1d : `~specutils.Spectrum1D`
        The loaded spectrum.
    """

    with fits.open(file_name, **kwargs) as hdulist:
        hdu = hdulist[ext]
        header = hdu.header
        table = Table.read(hdu)

    spectral_axis = table['spectral_axis'].value * table['spectral_axis'].unit

    flux = table['flux'].value * table['flux'].unit

    colnames = table.colnames

    if 'uncertainty' in colnames:
        uncertainty = StdDevUncertainty(table['uncertainty'].value)

    else:
        uncertainty = None
    
    if 'mask' in colnames:
        mask = table['mask'].value

    else:
        mask = None

    meta = {'header': header}
    
    spectrum1d = Spectrum1D(
        spectral_axis=spectral_axis, flux=flux, uncertainty=uncertainty, mask=mask, 
        meta=meta)

    return spectrum1d


def _AB_to_flux(wavelength, AB):
    """Convert AB magnitude to flux in [erg/cm2/s/A].

    Parameters
    ----------
    wavelength : `numpy.ndarray` or scalar
        wavelengths in [A].

    AB : `numpy.ndarray` or scalar
        AB magnitudes.

    returns
    -------
    flux : `numpy.ndarray`
        Flux in [erg/cm2/s/A].
    """

    c = 2.99792458e+10 # [cm/s]

    flux = 10**(-0.4 * AB) * 3631e-23 * c / wavelength**2 * 1e8 # [erg/cm2/s/A]

    return flux


def loadStandardSpectrum(standard):
    """Load standard spectrum.
    
    Parameters
    ----------
    standard : str
        Path to the file.

    Returns
    -------
    spectrum1d : `~specutils.Spectrum1D`
        Standard spectrum.
    
    bandpass : `~astropy.units.Quantity` or None
        Bandpass.
    """

    dir_library = os.path.realpath(
        os.path.join(
            os.path.split(os.path.realpath(__file__))[0], 'lib/onedstds'
        )
    )

    results = sorted(glob(os.path.join(dir_library, standard)))

    # All the standard spectrum should be named *.dat
    results = [result for result in results if result.endswith('.dat')]

    if not results:
        raise ValueError('No standard spectrum found.')

    elif len(results) > 1:
        raise ValueError(f'More than one standard spectrum found: {results}')

    else:
        path_to_standard = results[0]

        standard = os.path.split(path_to_standard)[1]

        library = os.path.split(os.path.split(path_to_standard)[0])[1]

        # todo: add libraries
        if library in ['blackbody', 'ctio']:
            raise ValueError('`blackbody` and `ctio` are not supported yet.')

        # Here no line is skipped because the first line starts with ``#``.
        spectrum = np.loadtxt(path_to_standard).T

        if library in ['bstdscal', 'ctiocal', 'ctionewcal', 'iidscal', 'irscal', 
                       'oke1990', 'redcal', 'spec16cal', 'spec50cal', 'spechayescal']:
            wavelength, AB, bandpass = spectrum
            flux = _AB_to_flux(wavelength, AB) * (u.erg / u.cm**2 / u.s /u.AA)
            bandpass = bandpass * u.AA

        # No bin for library `hststan`
        elif library == 'hststan':
            if standard.startswith('m'):
                wavelength, AB = spectrum
                flux = _AB_to_flux(wavelength, AB) * (u.erg / u.cm**2 / u.s /u.AA)

            else:
                # The third column is also flux but in [mJy]
                wavelength, flux, _ = spectrum
                flux = flux * 1e-16 * (u.erg / u.cm**2 / u.s /u.AA)

            bandpass = None

        elif library == 'okestan':
            if standard.startswith('m'):
                wavelength, AB, bandpass = spectrum
                flux = _AB_to_flux(wavelength, AB) * (u.erg / u.cm**2 / u.s /u.AA)

            else:
                wavelength, flux, _, bandpass = spectrum
                flux = flux * 1e-16 * (u.erg / u.cm**2 / u.s /u.AA)

            bandpass = bandpass * u.AA

        wavelength = wavelength * u.AA
        
        meta = {
            'header': {
                'FILENAME': standard
            }
        }

        spectrum1d = Spectrum1D(spectral_axis=wavelength, flux=flux, meta=meta)

        return spectrum1d, bandpass


def loadExtinctionCurve(extinct):
    """Load extinction curve.
    
    Parameters
    ----------
    extinct : str
        Path to the file. The current directory is searched prior to the library 
        directory.

    Returns
    -------
    spectrum1d : `~specutils.Spectrum1D`
        Extinction curve.
    """

    dir_current = os.getcwd()
    
    path_to_extinct = os.path.join(dir_current, extinct)

    try:

        wavelength, extinction = np.loadtxt(path_to_extinct).T

    except:

        dir_library = os.path.realpath(
            os.path.join(
                os.path.split(os.path.realpath(__file__))[0], 'lib/extinction'
            )
        )

        results = sorted(glob(os.path.join(dir_library, extinct)))

        # All the extinction curve should be named *.dat
        results = [result for result in results if result.endswith('.dat')]

        if not results:
            raise ValueError('No extinction curve found.')

        elif len(results) > 1:
            raise ValueError(f'More than one extinction curve found: {results}')

        else:
            path_to_extinct = results[0]

            wavelength, extinction = np.loadtxt(path_to_extinct).T
    
    finally:

        wavelength = wavelength * u.AA

        extinction = u.Quantity(extinction)

        meta = {
            'header': {
                'FILENAME': os.path.split(path_to_extinct)[1]
            }
        }

        spectrum1d = Spectrum1D(spectral_axis=wavelength, flux=extinction, meta=meta)

        return spectrum1d
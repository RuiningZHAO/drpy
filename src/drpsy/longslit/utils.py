from copy import deepcopy
from glob import glob
import os, warnings

# NumPy
import numpy as np
# Scipy
from scipy import signal
from scipy.optimize import curve_fit, OptimizeWarning
# AstroPy
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from astropy.nddata import StdDevUncertainty
# specutils
from specutils import Spectrum1D

from ..modeling.function import Gaussian1D
from ..validate import _validateSpectrum1D

__all__ = ['center1D', 'refinePeakBases', 'refinePeaks', 'loadSpectrum1D', 
           'loadStandardSpectrum']


def _center1D_Gaussian(x, y, initial_guess, threshold):
    """One dimensional centering using Gaussian fitting."""

    popt, _ = curve_fit(Gaussian1D, x, y, p0=initial_guess)

    amplitude, center, spread = popt

    if (amplitude > threshold) & (x[0] < center < x[-1]):
        return center

    else:
        raise RuntimeError('No feature fitted in the given interval.')

# todo: add Bartlett method.
def _center1D_Bartlett():
    """One dimensional centering using Bartlett window."""

    pass

# todo: validate threshold. add absorption. add Bartlett method.
def center1D(x, y, method, initial_guess, threshold=0):
    """1-dimensional centering.
    
    Parameters
    ----------
    x, y : array_like
        Input data.
    
    method : str
        Centering method.
    
    initial_guess : array_like
        Initial guess for the feature, and
        - if `Gaussian`, it should be an 1-dimensional array consist of amplitude, 
        center, spread, and offset.

    threshold : scalar, optional
        Feature.
    
    Returns
    -------
    center : scalar
        Center of the feature.
    """

    # Add method here
    _validateString(method, 'method', ['Gaussian', 'Bartlett'])
    
    if method == 'Gaussian':

        initial_guess = _validate1DArray(initial_guess, 'initial_guess', 4, False)
        
        center = _center1D_Gaussian(x, y, initial_guess, threshold)
        
    elif method == 'Bartlett':

        pass

    return center


def _refinePeakBases(peaks, left_bases, right_bases, n_peak, copy):
    """Refine left and right bases of peaks."""

    if copy:
        left_bases, right_bases = deepcopy(left_bases), deepcopy(right_bases)

    # Refine left base by comparing it with the right base of the former peak
    for i in range(1, n_peak):
        if right_bases[i - 1] < peaks[i]:
            left_bases[i] = np.max([left_bases[i], right_bases[i - 1]])

    # Refine right base by comparing it with the left base of the latter peak
    for i in range(0, n_peak - 1):
        if peaks[i] < left_bases[i + 1]:
            right_bases[i] = np.min([right_bases[i], left_bases[i + 1]])

    return left_bases, right_bases


def refinePeakBases(peaks, left_bases, right_bases):
    """Refine left and right bases of peaks.
    
    Parameters
    ----------
    peaks : array_like
        Peak locations in pixels.

    left_bases : array_like
        Left bases of peaks.

    right_bases : array_like
        Right bases of peaks.

    Returns
    -------
    left_bases, right_bases: `numpy.ndarray`
        Refined left and right bases of peaks.
    """
    
    try:
        peaks = np.array(peaks)
    except:
        raise TypeError(f'Invalid type `{type(peaks)}` for ``peaks``.')

    try:
        left_bases = np.array(left_bases)
    except:
        raise TypeError(f'Invalid type `{type(left_bases)}` for ``left_bases``.')

    try:
        right_bases = np.array(right_bases)
    except:
        raise TypeError(f'Invalid type `{type(right_bases)}` for ``right_bases``.')

    if peaks.shape[0] == left_bases.shape[0] == right_bases.shape[0]:
        n_peak = peaks.shape[0]

    else:
        raise ValueError(
            'Lengths of ``peaks``, ``left_bases`` and ``right_bases`` are not equal.')

    if np.all((peaks - left_bases) > 0) & np.all((right_bases - peaks) > 0):
        return _refinePeakBases(peaks, left_bases, right_bases, n_peak, copy=False)

    else:
        raise ValueError(
            'Peaks should locate between corresponding left and right bases.')


def _refinePeaks(spectrum, peaks, heights, left_bases, right_bases, tolerance):
    """Refine peak locations."""

    widths = right_bases - left_bases

    left_bases = np.round(left_bases).astype(int)
    right_bases = np.round(right_bases).astype(int) + 1

    index = np.arange(spectrum.shape[0])

    with warnings.catch_warnings():

        warnings.simplefilter('error', OptimizeWarning)

        refined_index = list()
        refined_peaks = list()

        for i in range(peaks.shape[0]):

            initial_guess = heights[i], peaks[i], widths[i]

            x = index[left_bases[i]:right_bases[i]]
            y = spectrum[left_bases[i]:right_bases[i]]

            try:
                center = _center1D_Gaussian(x, y, initial_guess, 0)
                refined_index.append(i); refined_peaks.append(center)

            except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
                pass

    refined_index = np.array(refined_index)
    refined_peaks = np.array(refined_peaks)

    # Mask peaks that are close from each other
    mask = np.isclose(refined_peaks[:-1], refined_peaks[1:], rtol=0, atol=tolerance)
    mask = np.hstack([False, mask])

    refined_index = refined_index[~mask]
    refined_peaks = refined_peaks[~mask]

    return refined_peaks, refined_index


def refinePeaks(spectrum, peaks, properties, tolerance=1):
    """Refine peak locations in a spectrum from a set of initial estimates. This 
    function attempts to fit a Gaussian to each peak in the provided list. It returns a 
    list of sub-pixel refined peaks. If two peaks are very close, they can be refined 
    to the same location. In this case only one of the peaks will be returned - i.e. 
    this function will return a unique set of peak locations.

    Parameters
    ----------
    spectrum : array_like
        Input spectrum (intensities).

    peaks : array_like
        Peak locations in pixels.

    properties : dict
        Peak properties dictionary returned by `~scipy.signal.find_peaks`. Keywords 
        `peak_heights`, `left_bases` and `right_bases` are needed in Gaussian fittings. 
        They will be recalculated if not available in the dictionary.
    
    tolerance : scalar, optional
        If the distance between two peaks < ``tolerance``, they are refined to the same 
        location.
    
    refine_bases : bool, optional
        Whether to refine left and right bases first.

    Returns
    -------
    refined_peaks: `numpy.ndarray`
        Refined peak locations in pixels.
    
    refined_properties: dict
        Properties dictionary of the refined peaks.
    
    refined_index : `numpy.ndarray`
        Index of the refined peaks in the input peak array.
    """

    try:
        spectrum = np.array(spectrum, dtype=float)
    except:
        raise TypeError(f'Invalid type `{type(spectrum)}` for ``spectrum``.')

    try:
        peaks = np.array(peaks, dtype=float)
    except:
        raise TypeError(f'Invalid type `{type(peaks)}` for ``peaks``.')

    if 'peak_heights' not in properties:
        heights = spectrum[np.round(peaks).astype(int)]
    else:
        try:
            heights = np.array(properties['peak_heights'], dtype=float)
        except:
            raise TypeError(
                f"Invalid type `{type(properties['peak_heights'])}` for ``heights``.")

    if ('left_bases' not in properties) or ('right_bases' not in properties):
        _, left_bases, right_bases = signal.peak_prominences(
            x=spectrum, peaks=np.round(peaks).astype(int), wlen=None)
    else:
        try:
            left_bases = np.array(properties['left_bases'], dtype=float)
        except:
            raise TypeError(
                f"Invalid type `{type(properties['left_bases'])}` for "
                "``left_bases``.")
        try:
            right_bases = np.array(properties['right_bases'], dtype=float)
        except:
            raise TypeError(
                f"Invalid type `{type(properties['right_bases'])}` for "
                "``right_bases``.")

    n_peak = peaks.shape[0]

    if not (heights.shape[0] == left_bases.shape[0] == right_bases.shape[0] == n_peak):
        raise ValueError(
            'Lengths of ``peaks``, ``heights``, ``left_bases`` and ``right_bases`` '
            'are not equal.')

    if not (np.all((peaks - left_bases) > 0) & np.all((right_bases - peaks) > 0)):
        raise ValueError(
            'Peaks should locate between corresponding left and right bases.')

    left_bases, right_bases = _refinePeakBases(
        peaks, left_bases, right_bases, n_peak=n_peak, copy=True)

    refined_peaks, refined_index = _refinePeaks(
        spectrum, peaks, heights, left_bases, right_bases, tolerance)

    refined_properties = deepcopy(properties)
    for key, val in refined_properties.items():
        refined_properties[key] = val[refined_index]
    refined_properties['left_bases'] = left_bases[refined_index]
    refined_properties['right_bases'] = right_bases[refined_index]

    return refined_peaks, refined_properties, refined_index


def _Spectrum1D_to_hdu(spectrum1d):
    """Convert a `~specutils.Spectrum1D` object to hdu."""

    # spectral_axis
    unit = spectrum1d.spectral_axis.unit.to_string('fits')
    array = spectrum1d.spectral_axis.value
    spectral_axis = fits.Column(
        name='spectral_axis', format='D', unit=unit, array=array)
    
    # flux
    unit = spectrum1d.flux.unit.to_string('fits')
    array = spectrum1d.flux.value

    ndim = np.ndim(array)

    if ndim <= 1:
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

    hdu = fits.BinTableHDU.from_columns(columns)

    return hdu


# todo: add additional header
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

    hdu = _Spectrum1D_to_hdu(new_spectrum1d)
    
#     # header
#     if 'header' in new_spectrum1d.meta:
#         header = new_spectrum1d.meta['header']
#         for (key, value), comment in zip(header.items(), header.comments):
#             if key not in hdu.header:
#                 hdu.header[key] = (value, comment)
#     hdu.header['MODIFIED'] = (
#         '{}'.format(Time.now().to_value('iso', subfmt='date_hm')), 'last modified')

    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])

    hdulist.writeto(file_name, overwrite=True)

    return None


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
        uncertainty = StdDevUncertainty(table['uncertainty'])
    else:
        uncertainty = None
    
    if 'mask' in colnames:
        mask = table['mask']
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

# todo: remove .ipynb_checkpoints
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
            os.path.split(os.path.realpath(__file__))[0], '../lib/onedstds'
        )
    )

    results = sorted(glob(os.path.join(dir_library, standard)))

    # All the standard spectrum should be named *.dat
    results = [result for result in results if result.endswith('.dat')]

    if not results:
        raise ValueError('No standard spectrum found.')

    elif len(results) > 1:
        for result in results:
            print(result)
        raise ValueError('More than one standard spectrum found.')

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
                os.path.split(os.path.realpath(__file__))[0], '../lib/extinction'
            )
        )

        results = sorted(glob(os.path.join(dir_library, extinct)))

        # All the extinction curve should be named *.dat
        results = [result for result in results if result.endswith('.dat')]

        if not results:
            raise ValueError('No extinction curve found.')

        elif len(results) > 1:
            for result in results:
                print(result)
            raise ValueError('More than one extinction curve found.')

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
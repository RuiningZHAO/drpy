import warnings
from copy import deepcopy

# NumPy
import numpy as np
# Scipy
from scipy import signal
from scipy.optimize import curve_fit, OptimizeWarning
# drpsy
from drpsy.modeling.function import Gaussian1D

__all__ = ['center1D', 'refinePeakBases', 'refinePeaks']


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
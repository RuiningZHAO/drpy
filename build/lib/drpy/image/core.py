import warnings
from copy import deepcopy

# AstroPy
from astropy.time import Time
# ccdproc
from ccdproc.utils.slices import slice_from_string
# drpy
from drpy.validate import _validate1DArray, _validateCCDList

__all__ = ['concatenate']


def concatenate(ccdlist, fits_section, scale=None):
    """Concatenate two frames.

    The region defined by ``fits_section`` of the second frame is replaced by that in 
    the first frame. Usually this is used to concatenate flat-fields or arc spectra of 
    different exposure times. To concatenate multiple frames, call this function 
    repeatedly.

    Parameters
    ----------
    ccdlist : Iterable
        Iterable object containing two 2-dimensional frames.

    fits_section : str
        Region of the second frame to be replaced by the counterpart of the first frame.
    
    scale : scalar, array-like or `None`, optional
        Scaling factor. Frames are multiplied by scaling factor prior to concatenation.
        Default is `None`.

    Returns
    -------
    nccd : `~astropy.nddata.CCDData`
        The concatenated frame.
    """

    ccdlist = _validateCCDList(ccdlist, 2, 2)
    n_image = len(ccdlist)
    if n_image > 2:
        warnings.warn(
            '``ccdlist`` contains more than two frames. Extra frames will be ignored.', 
            RuntimeWarning)

    python_slice = slice_from_string(fits_section, fits_convention=True)

    if scale is None:
        scale = _validate1DArray(1, 'scale', 2, True)

    else:
        scale = _validate1DArray(scale, 'scale', 2, True)

    data_arr = ccdlist[1].data * scale[1]
    data_arr[python_slice] = ccdlist[0].data[python_slice] * scale[0]

    if (ccdlist[0].uncertainty is None) | (ccdlist[1].uncertainty is None):
        uncertainty_arr = None

    else:
        uncertainty_arr = scale[1] * ccdlist[1].uncertainty.array
        uncertainty_arr[python_slice] = (
            ccdlist[0].uncertainty.array[python_slice] * scale[0])

    if (ccdlist[0].mask is None) | (ccdlist[1].mask is None):
        mask_arr = None

    else:
        mask_arr = deepcopy(ccdlist[1].mask)
        mask_arr[python_slice] = ccdlist[0].mask[python_slice]

    nccd = ccdlist[0].copy()

    nccd.data = data_arr

    if uncertainty_arr is None:
        nccd.uncertainty = None

    else:
        # Here ``nccd.uncertainty`` is definitely not `None`.
        nccd.uncertainty.array = uncertainty_arr

    nccd.mask = mask_arr

    nccd.header['CONCATEN'] = (
        '{} Data section {} is from the first frame.'.format(
            Time.now().to_value('iso', subfmt='date_hm'), fits_section)
    )

    return nccd
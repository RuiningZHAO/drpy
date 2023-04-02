# NumPy
import numpy as np
# AstroPy
from astropy.table import Table
from astropy.nddata import CCDData

from .validate import (_validateBool, _validateCCDData, _validateCCDList)

__all__ = ['imstatistics']


def imstatistics(ccdlist, verbose=False):
    """Image statistics.
    
    Parameters
    ----------
    ccdlist : Iterable or `~astropy.nddata.CCDData` or `~numpy.ndarray`
        A 2-dimensional image or an iterable object consist of 2-dimensional 
        images, the statistics of which are calculated.

    verbose : bool, optional
        If `True`, statistics table is printted before return.
        Default is `False`.
    
    Returns
    -------
    stats : `~astropy.table.table.Table`
        Statistics table.
    """

    if not isinstance(ccdlist, (np.ndarray, CCDData)):
        nccdlist = _validateCCDList(ccdlist, 1, 2) # dimension is set to 2
    else:
        nccdlist = [_validateCCDData(ccdlist, 'ccd')]
    
    n_image = len(nccdlist)

    # Construct table
    rows = list()
    for i in range(n_image):
        
        ccd = nccdlist[i]
        
        if 'FILENAME' in ccd.header:
            file_name = ccd.header['FILENAME']
        else:
            file_name = f'image {i + 1}'

        # !!! mask is effective here !!!
        rows.append(
            [file_name, ccd.shape, np.mean(ccd.data), np.std(ccd.data, ddof=1), 
             np.min(ccd.data), np.max(ccd.data)])

    stats = Table(rows=rows, names=['IMAGE', 'SHAPE', 'MEAN', 'STDDEV', 'MIN', 'MAX'])
    
    # Format
    for colname in ['MEAN', 'STDDEV', 'MIN', 'MAX']:
        stats[colname].format = "g"

    _validateBool(verbose, 'verbose')

    if verbose: stats.pprint_all()

    return stats
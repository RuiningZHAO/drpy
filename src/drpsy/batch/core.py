import os, textwrap, warnings
from copy import deepcopy

# NumPy
import numpy as np
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.nddata import CCDData
# ccdproc
from ccdproc import (trim_image, combine, subtract_bias, create_deviation, 
                     flat_correct, cosmicray_lacosmic, cosmicray_median)
# drpsy
from drpsy import conf
from drpsy.utils import imstatistics
from drpsy.validate import _validateBool, _validateString

# Configurations
unit_ccddata = u.Unit(conf.unit_ccddata)

__all__ = ['CCDDataList']


class CCDDataList:
    """A class for batch processing."""

    @classmethod
    def read(cls, file_list, hdu=1, unit=unit_ccddata):
        """Construct a `CCDDataList` object from a list of files.
        
        Parameters
        ----------
        file_list : array_like
            List of files.

        hdu : `int`, optional
            FITS extension from which a `CCDDataList` object should be initialized.
            Default is `1`.

        unit : `~astropy.units.Unit` or `str`, optional
            Unit of the image data.
            Default is `~astropy.units.adu`.

        Returns
        -------
        ccddatalist : `CCDDataList`
            Constructed `CCDDataList` object.
        """
        
        print(unit)

        ccddatalist = list()
        for file_name in file_list:
            # Load from file
            ccd = CCDData.read(file_name, hdu=hdu, unit=unit)
            # Add or rewrite keyword `FILENAME` (used in `self.statistics`).
            ccd.header['FILENAME'] = os.path.split(file_name)[1]
            ccddatalist.append(ccd)

        ccddatalist = cls(ccddatalist)

        return ccddatalist


    def __init__(self, ccddatalist):
        """Construct a `CCDDataList` object.
        
        The size of each image should be the same. (This is not checked in the 
        current version.)

        Parameters
        ----------
        ccddatalist : Iterable
            List of `~astropy.nddata.CCDData` objects.
        """

        for ccd in ccddatalist:
            if not isinstance(ccd, CCDData):
                raise TypeError(
                    f'Invalid type `{type(ccd)}` for an item of `CCDDataList` object. '
                    '`~astropy.nddata.CCDData` is expected.')

        self._ccddatalist = deepcopy(ccddatalist)   


    def trim(self, row_range, col_range):
        """Trim images.

        Parameters
        ----------
        row_range : array_like
            Row range (python style).

        col_range : array_like
            Column range (python style).
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            Trimmed images.
        """
        
        # Custom keywords
        keywords_dict = {
            'WCSDIM': 2,
            'LTM1_1': 1.,
            'LTM2_2': 1.,
            'LTV1': -col_range[0],
            'LTV2': -row_range[0],
            'WAT0_001': 'system=physical',
            'WAT1_001': 'wtype=linear',
            'WAT2_001': 'wtype=linear',
            'TRIM': '{} Trim data section is [{}:{},{}:{}]'.format(
                Time.now().to_value('iso', subfmt='date_hm'), (col_range[0] + 1), 
                col_range[1], (row_range[0] + 1), row_range[1]),
            'CCDSEC': '[{}:{},{}:{}]'.format(
                (col_range[0] + 1), col_range[1], (row_range[0] + 1), row_range[1]),
            'BIASSEC': '[1:{},1:{}]'.format(
                (col_range[1] - col_range[0]), (row_range[1] - row_range[0])),
        }

        ccddatalist = list()
        for ccd in self._ccddatalist:
            ccddatalist.append(
                trim_image(
                    ccd=ccd[row_range[0]:row_range[1], col_range[0]:col_range[1]], 
                    add_keyword=keywords_dict)
            )
        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def combine(self, method, **kwargs):
        """Combine images.
        
        Parameters
        ----------
        method : `str`
            `average`, `median`, or `sum`.
        
        Returns
        -------
        combined_ccd : `~astropy.nddata.CCDData`
            Combined frame.
        """
        
        combined_ccd = combine(
            img_list=self._ccddatalist, method=method, **kwargs)

        combined_ccd.header = self._ccddatalist[0].header
        combined_ccd.header['NCOMBINE'] = (self.__len__(), 'number of frames combined')
        
        return combined_ccd


    def subtract(self, master):
        """Subtract ``master`` frame from the `CCDDataList` object.

        Parameters
        ----------
        master : `~astropy.nddata.CCDData`
            Image to be subtracted from the `CCDDataList` object. Usually a master bias.

        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with ``master`` subtracted.
        """

        if not isinstance(master, CCDData):
            raise TypeError(
                f'Invalid type `{type(master)}` for ``master``. '
                '`~astropy.nddata.CCDData` is expected.')

        keywords_dict = {
            'ZEROCOR': '{} A level of {} is subtracted'.format(
                Time.now().to_value('iso', subfmt='date_hm'), round(master.data.mean()))
        }

        ccddatalist = list()
        for ccd in self._ccddatalist:
            ccddatalist.append(
                subtract_bias(ccd=ccd, master=master, add_keyword=keywords_dict)
            )
        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist

    # todo: use min_value to get rid of negative values ???
    def divide(self, master):
        """Divide ``master`` from the `CCDDataList` object.

        Parameters
        ----------
        master : `~astropy.nddata.CCDData`
            Image to be divided from the `CCDDataList` object. Usually a master flat.

        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with ``master`` divided.
        """

        if not isinstance(master, CCDData):
            raise TypeError(
                f'Invalid type `{type(master)}` for ``master``. '
                '`~astropy.nddata.CCDData` is expected.')

        keywords_dict = {
            'FLATCOR': '{} flat-fielded'.format(
                Time.now().to_value('iso', subfmt='date_hm'))
        }

        ccddatalist = list()
        for ccd in self._ccddatalist:
            ccddatalist.append(
                flat_correct(
                    ccd=ccd, flat=master, norm_value=1, add_keyword=keywords_dict)
            )
        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def create_deviation(self, gain, readnoise, **kwargs):
        """Create deviation.

        Parameters
        ----------
        gain : scalar
            Gain in [e- / adu].

        noise : scalar
            Read noise in [e-].

        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with uncertainty assigned.
        """

        ccddatalist = list()
        for ccd in self._ccddatalist:
            # the pre-assigned uncertainties will be replaced by new ones 
            ccddatalist.append(
                create_deviation(
                    ccd_data=ccd, gain=(gain * u.electron / u.adu), 
                    readnoise=(readnoise * u.electron), **kwargs)
            )
        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def cosmicray(self, method, use_mask=False, gain=None, readnoise=None, 
                  gain_apply=False, **kwargs):
        """Remove cosmic rays.
        
        Parameters
        ----------
        method : str
            `Laplacian` or `median`. Cosmic ray removal method.

        use_mask : bool, optional
            If `True`, the attributed mask frames are used in identifying cosmic ray 
            pixels. (only useful for `Laplacian` method)
            Default is `False`.

        gain : scalar or `None`, optional
            Gain in [e- / adu].

        readnoise : scalar or `None`, optional
            Read noise in [e-].

        gain_apply : bool, optional
            If `True`, return gain-corrected data, with correct units, otherwise do not 
            gain-correct the data. `False` is recommended.

        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with cosmic rays removed.
        """

        _validateBool(use_mask, 'use_mask')

        _validateString(method, 'method', ['Laplacian', 'median'])

        if method == 'Laplacian':
            keywords_dict = {
                'COSMICRA': '{} Laplacian Edge Detection'.format(
                    Time.now().to_value('iso', subfmt='date_hm'))
            }
        elif method == 'median':
            keywords_dict = {
                'COSMICRA': '{} median with sigma clipping'.format(
                    Time.now().to_value('iso', subfmt='date_hm'))
            }

        ccddatalist = list()
        for ccd in self._ccddatalist:

            nccd = ccd.copy()

            if use_mask:
                if nccd.mask is None:
                    warnings.warn(
                        'The input mask is unavailable. All set unmasked.', 
                        RuntimeWarning)
                    nccd.mask = np.zeros_like(nccd.data, dtype=bool)
            else:
                nccd.mask = None
            
            if method == 'Laplacian':

                ccd_cosmicray = cosmicray_lacosmic(
                    ccd=nccd, gain=gain, readnoise=readnoise, gain_apply=gain_apply, 
                    **kwargs)

            elif method == 'median':

                ccd_cosmicray = cosmicray_median(ccd=nccd, **kwargs)

            ccd_cosmicray.unit = nccd.unit

            for key, value in keywords_dict.items():
                ccd_cosmicray.header[key] = value

            ccddatalist.append(ccd_cosmicray)

        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def statistics(self, verbose=False):
        """Statistics.

        Parameters
        ----------
        verbose : bool, optional
            If `True`, statistics table is printted before return.
            Default is `False`.

        Returns
        -------
        out : `~astropy.table.table.Table`
            Statistics table.
        """

        return imstatistics(self._ccddatalist, verbose=verbose)


    def to_list(self):
        """Convert to `list` object."""
        return self._ccddatalist


    def copy(self):
        """Return a copy of itself."""
        return self.__class__(self._ccddatalist)


    def __getitem__(self, item):

        if isinstance(item, (int, np.integer)):
            return self._ccddatalist[item]

        elif isinstance(item, slice):
            return self.__class__(self._ccddatalist[item])

        elif isinstance(item, (np.ndarray, list)):

            if isinstance(item, list):
                item = np.array(item)

            if item.ndim > 1:
                raise IndexError(
                    f'`CCDDataList` is 1-dimensional, but {item.ndim} were indexed.')

            else:
                if item.dtype == bool:

                    if item.size != self.__len__():
                        raise IndexError(
                            'Boolean index did not match the size of `CCDDataList`.')

                    else:
                        item = np.where(item)[0]

                ccddatalist = list()
                for i in item:
                    ccddatalist.append(self._ccddatalist[i])

                return self.__class__(ccddatalist)

        else:
            raise IndexError(
                f'Invalid type {type(item)} for `CCDDataList` item access.')


    def __sub__(self, master):
        """The same as `self.subtract`."""
        return self.subtract(master)


    def __truediv__(self, master):
        """The same as `self.divide`."""
        return self.divide(master)


    def __repr__(self):

        if len(self._ccddatalist) == 0:
            core_string = ''

        elif len(self._ccddatalist) == 1:
            core_string = f'{self._ccddatalist[0]}'

        elif len(self._ccddatalist) == 2:
            core_string = f'{self._ccddatalist[0]},\n{self._ccddatalist[-1]}'

        elif len(self._ccddatalist) > 2:
            core_string = f'{self._ccddatalist[0]},\n...,\n{self._ccddatalist[-1]}'

        return (
            'CCDDataList([' + textwrap.indent(core_string, ' ' * 13)[13:] + '], length='
            f'{self.__len__()})'
        )


    def __len__(self):
        return len(self._ccddatalist)


    def __iter__(self):
        return iter(self._ccddatalist)
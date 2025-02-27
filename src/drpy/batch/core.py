import os, textwrap, warnings
from copy import deepcopy

# NumPy
import numpy as np
# AstroPy
import astropy.units as u
from astropy.time import Time
from astropy.nddata import CCDData
# ccdproc
import ccdproc
# drpy
from drpy import conf
from drpy.utils import imstatistics
from drpy.validate import _validateBool

# Configurations
unit_ccddata = u.Unit(conf.unit_ccddata)
dtype_ccddata = np.dtype(conf.dtype_ccddata)

__all__ = ['CCDDataList']


class CCDDataList:
    """A class for batch processing."""

    @classmethod
    def read(cls, file_list, hdu=1, unit=unit_ccddata, dtype=dtype_ccddata):
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

        dtype : `type` or `str`, optional
            Data type of the image data.
            Default is `~numpy.float32`.

        Returns
        -------
        ccddatalist : `CCDDataList`
            Constructed `CCDDataList` object.
        """

        ccddatalist = list()
        for file_name in file_list:

            # Load from file
            ccd = CCDData.read(file_name, hdu=hdu, unit=unit)

            # Convert to specified data type
            if ccd.data is not None:
                ccd.data = ccd.data.astype(dtype)

            if ccd.uncertainty is not None:
                ccd.uncertainty.arraye = ccd.uncertainty.array.astype(dtype)

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


    def apply_over_ccd(self, func, *args, **kwargs):
        """Apply a function repeatedly over all the images.
        
        Parameters
        ----------
        func : function
            This function must take the to-be-processed image as its first argument.
        
        *args : tuple, optional
            Positional arguments passed through to ``func``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``func``.
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            Processed images.
        """

        ccddatalist = list()

        for ccd in self._ccddatalist:

            ccddatalist.append(func(ccd, *args, **kwargs))

        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def transform_image(self, *args, **kwargs):
        """Transform images.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.transform_image``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.transform_image``.
        """

        return self.apply_over_ccd(ccdproc.transform_image, *args, **kwargs)

    
    def trim_image(self, *args, **kwargs):
        """Trim images to the dimensions indicated.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.trim_image``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.trim_image``.
        """

        return self.apply_over_ccd(ccdproc.trim_image, *args, **kwargs)


    def wcs_project(self, *args, **kwargs):
        """Project images onto a target WCS.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.wcs_project``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.wcs_project``.
        """

        return self.apply_over_ccd(ccdproc.wcs_project, *args, **kwargs)


    def subtract_bias(self, master, **kwargs):
        """Subtract master bias from images.
        
        Parameters
        ----------
        master : `~astropy.nddata.CCDData`
            Master image to be subtracted from input images.
        
        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.subtract_bias``.
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with bias subtracted.
        """

        if 'add_keyword' not in kwargs:

            keywords_dict = {
                'ZEROCOR': '{} A level of {} is subtracted'.format(
                    Time.now().to_value('iso', subfmt='date_hm'), 
                    round(np.nanmean(master.data))
                )
            }

            ccddatalist =  self.apply_over_ccd(
                ccdproc.subtract_bias, master, add_keyword=keywords_dict, **kwargs)

        else:

            ccddatalist = self.apply_over_ccd(ccdproc.subtract_bias, master, **kwargs)
        
        return ccddatalist


    def subtract_dark(self, *args, **kwargs):
        """Subtract dark current from images.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.subtract_dark``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.subtract_dark``.
        """

        return self.apply_over_ccd(ccdproc.subtract_dark, *args, **kwargs)


    def subtract_overscan(self, *args, **kwargs):
        """Subtract the overscan region from images.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.subtract_overscan``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.subtract_overscan``.
        """

        return self.apply_over_ccd(ccdproc.subtract_overscan, *args, **kwargs)


    def flat_correct(self, flat, min_value=None, norm_value=None, **kwargs):
        """Correct images for flat fielding.
        
        Parameters
        ----------
        flat : `~astropy.nddata.CCDData`
            Flat field to apply to the data.

        min_value : float or `None`, optional
            Minimum value for flat field. The value can either be `None` and no minimum 
            value is applied to the flat or specified by a float which will replace all 
            values in the flat by the min_value.
            Default is `None`.

        norm_value : float or `None`, optional
            If not `None`, normalize flat field by this argument rather than the mean 
            of the image. This allows fixing several different flat fields to have the 
            same scale. If this value is negative or `0`, a `ValueError` is raised. 
            Default is `None`.
        
        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.flat_correct``.
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with flat corrected.
        """

        if 'add_keyword' not in kwargs:

            keywords_dict = {
                'FLATCOR': '{} flat-fielded'.format(
                    Time.now().to_value('iso', subfmt='date_hm'))
            }

            ccddatalist = self.apply_over_ccd(
                ccdproc.flat_correct, flat, min_value, norm_value, 
                add_keyword=keywords_dict, **kwargs)

        else:

            ccddatalist = self.apply_over_ccd(
                ccdproc.flat_correct, flat, min_value, norm_value, **kwargs)

        return ccddatalist


    def create_deviation(self, *args, **kwargs):
        """Create uncertainty frames.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.create_deviation``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.create_deviation``.
        """

        return self.apply_over_ccd(ccdproc.create_deviation, *args, **kwargs)


    def gain_correct(self, *args, **kwargs):
        """Correct gain in images.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.gain_correct``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.gain_correct``.
        """

        return self.apply_over_ccd(ccdproc.gain_correct, *args, **kwargs)


    def ccd_process(self, *args, **kwargs):
        """Perform basic processing on ccd data.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.ccd_process``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.ccd_process``.
        """

        return self.apply_over_ccd(ccdproc.ccd_process, *args, **kwargs)


    def combine(self, *args, **kwargs):
        """Combine images.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.combine``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.combine``.
        
        Returns
        -------
        combined_ccd : `~astropy.nddata.CCDData`
            Combined image.
        """
        
        combined_ccd = ccdproc.combine(img_list=self._ccddatalist, *args, **kwargs)

        combined_ccd.header = self._ccddatalist[0].header
        combined_ccd.header['NCOMBINE'] = (self.__len__(), 'number of images combined')
        
        return combined_ccd


    def add(self, *args, **kwargs):
        """Perform addition.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~astropy.nddata.CCDData.add``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~astropy.nddata.CCDData.add``.
        """

        return self.apply_over_ccd(CCDData.add, *args, **kwargs)


    def subtract(self, *args, **kwargs):
        """Perform subtraction.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~astropy.nddata.CCDData.subtract``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~astropy.nddata.CCDData.subtract``.
        """

        return self.apply_over_ccd(CCDData.subtract, *args, **kwargs)


    def multiply(self, *args, **kwargs):
        """Perform multiplication.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~astropy.nddata.CCDData.multiply``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~astropy.nddata.CCDData.multiply``.
        """

        return self.apply_over_ccd(CCDData.multiply, *args, **kwargs)


    def divide(self, *args, **kwargs):
        """Perform division.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~astropy.nddata.CCDData.divide``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~astropy.nddata.CCDData.divide``.
        """

        return self.apply_over_ccd(CCDData.divide, *args, **kwargs)


    def cosmicray_lacosmic(self, use_mask=False, *args, **kwargs):
        """Remove cosmic rays.
        
        Parameters
        ----------
        use_mask : bool, optional
            If `True`, the attributed mask frames are used in identifying cosmic ray 
            pixels.
            Default is `False`.
        
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.cosmicray_lacosmic``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.cosmicray_lacosmic``.
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with cosmic rays removed.
        """

        _validateBool(use_mask, 'use_mask')

        keywords_dict = {
            'COSMICRA': '{} Laplacian Edge Detection'.format(
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

            # Of dtype float32
            ccd_cosmicray_removed = ccdproc.cosmicray_lacosmic(
                ccd=nccd, *args, **kwargs)

            for key, value in keywords_dict.items():
                ccd_cosmicray_removed.header[key] = value

            ccddatalist.append(ccd_cosmicray_removed)

        ccddatalist = self.__class__(ccddatalist)

        return ccddatalist


    def cosmicray_median(self, *args, **kwargs):
        """Remove cosmic rays.
        
        Parameters
        ----------
        *args : tuple, optional
            Positional arguments passed through to ``~ccdproc.cosmicray_median``.

        **kwargs : dict, optional
            Keyword arguments passed through to ``~ccdproc.cosmicray_median``.
        
        Returns
        -------
        ccddatalist : `CCDDataList`
            `CCDDataList` object with cosmic rays removed.
        """

        keywords_dict = {
            'COSMICRA': '{} median with sigma clipping'.format(
                Time.now().to_value('iso', subfmt='date_hm'))
        }

        ccddatalist = list()

        for ccd in self._ccddatalist:

            ccd_cosmicray_removed = ccdproc.cosmicray_median(ccd=ccd, *args, **kwargs)

            for key, value in keywords_dict.items():

                ccd_cosmicray_removed.header[key] = value

            ccddatalist.append(ccd_cosmicray_removed)

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
        return self.copy()._ccddatalist


    def copy(self):
        """Return a copy of itself."""
        return self.__class__(self._ccddatalist)


    def __add__(self, operand2):
        """The same as `self.add`."""
        return self.add(operand2, handle_meta='first_found')


    def __sub__(self, operand2):
        """The same as `self.subtract`."""
        return self.subtract(operand2, handle_meta='first_found')


    def __mul__(self, operand2):
        """The same as `self.multiply`"""
        return self.multiply(operand2, handle_meta='first_found')


    def __truediv__(self, operand2):
        """The same as `self.divide`."""
        return self.divide(operand2, handle_meta='first_found')


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
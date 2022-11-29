import warnings

# NumPy
import numpy as np
# SciPy
from scipy.optimize import curve_fit, OptimizeWarning
# matplotlib
import matplotlib.pyplot as plt
# AstroPy
from astropy.nddata import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm
# photutils
from photutils import CircularAperture, RectangularAperture
from photutils.background import Background2D
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

from drpsy import conf
from drpsy.modeling.function import CircularGaussian
from drpsy.plotting import _plot2d
from drpsy.validate import (_validateString, _validateBool, _validateRange, 
                            _validateInteger, _validateCCD, _validatePath)

__all__ = ['getFWHM']

# todo: deal with NaNs in Gaussian fitting.
def getFWHM(ccd, box_size, n_sigma, aper_radius, saturation, n_peak, use_mask=False, 
            title='FWHM', show=conf.show, save=conf.save, path=conf.path, **kwargs):
    """Estimate mean FWHM of the sources in the input image.

    The background of the input image is first modeled by a 2-dimensional background 
    estimator, and then subtracted from the input image. Detect sources in the 
    background subtracted image. Fit circular Gaussian profiles to the detected 
    sources.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData` or `~numpy.ndarray`
        Input image.

    box_size : int or array_like
        The box size along each axis. If ``box_size`` is a scalar then a square box of 
        size ``box_size`` will be used. If ``box_size`` has two elements, they must be 
        in (ny, nx) order. (for details see https://photutils.readthedocs.io/en/stable/
        api/photutils.background.Background2D.html)

    n_sigma : scalar
        The detection threshold is defined as ``n_sigma`` * background RMS.

    aper_radius : int
        Radius of a source in the input image. It is recommended to set comparable to 
        or slightly larger than the FWHM.

    saturation : scalar
        Saturation level of pixels.

    n_peak : int
        Number of peaks. When number of peaks exceeds ``n_peak``, the peaks with the 
        highest peak intensities are used in determining FWHM.

    use_mask : bool, optional
        If `True` and a mask array is attributed to ``ccd``, the masked pixels are 
        ignored in the background estimation and source detection. 
        Default is `False`.

    Returns
    -------
    fwhm : scalar
        Mean FWHM of the detected sources.

    fwhm_err : scalar
        Standard deviation of FWHM.
    """

    _validateInteger(aper_radius, 'aper_radius', (1, None), (True, None))

    # Assumes that ``aper_radius`` ~ FWHM ~ 2.3σ. Thus, for a circular Gaussian 
    # profile, almost 100% of the energy are concentrated in the aperture.
    aper_size = 2 * aper_radius + 1

    _validateBool(use_mask, 'use_mask')

    nccd, data_arr, _, mask_arr = _validateCCD(ccd, 'ccd', False, use_mask, False)

    # Background estimation
    bkg = Background2D(data=data_arr, box_size=box_size, mask=mask_arr, **kwargs)
    bkg_value = bkg.background
    bkgrms_value = bkg.background_rms

    _validateRange(n_sigma, 'n_sigma', (0, None), (False, None))

    # Source detection
    threshold = n_sigma * bkgrms_value
    data_arr_bkgsb = data_arr - bkg_value
    peak_table = find_peaks(
        data=data_arr_bkgsb, threshold=threshold, box_size=aper_size, footprint=None, 
        mask=mask_arr, border_width=(2 * aper_size - 1), centroid_func=centroid_com, 
        error=None, wcs=None)
    peak_table.sort('peak_value')
    peak_table.reverse()

    _validateRange(saturation, 'saturation', (0, None), (False, None))

    # Mask saturated sources
    isSaturated = peak_table['peak_value'].data >= saturation

    # Cutout size = 4 * ``aper_size`` - 3 = 8 * ``aper_radius`` + 1
    size = 4 * aper_size - 3

    # Mask sources too close to each other
    x, y = peak_table['x_centroid'].data, peak_table['y_centroid'].data
    isClose = np.zeros_like(x, dtype=bool)
    for i in range(x.shape[0]):
        if not isClose[i]:
            for j in range(i + 1, x.shape[0]):
                dist_sq = (x[j] - x[i])**2 + (y[j] - y[i])**2
                if dist_sq < size**2 / 2:
                    isClose[i], isClose[j] = True, True
    
    peak_table = peak_table[(~isSaturated) & (~isClose)]

    _validateInteger(n_peak, 'n_peak', (1, None), (True, None))

    # Remove redundant peaks
    peak_table = peak_table[:n_peak]

    # Update peak number (may become smaller)
    n_peak = len(peak_table)

    # Centroid
    x, y = peak_table['x_centroid'].data, peak_table['y_centroid'].data

    # X & Y grids
    Y, X = np.mgrid[-(size // 2):(size // 2 + 1), -(size // 2):(size // 2 + 1)]

    xy = (X.flatten(), Y.flatten())

    xy_min, xy_max = xy[0][0], xy[0][-1]

    position_arr = np.zeros((n_peak, 2)) + np.nan
    fwhm_arr = np.zeros(n_peak) + np.nan
    cutout2d_arr = np.zeros((n_peak, size, size))

    with warnings.catch_warnings():

        warnings.simplefilter('error', OptimizeWarning)

        for i in range(n_peak):

            # Cutout
            cutout2d = Cutout2D(
                data=data_arr_bkgsb, position=(x[i], y[i]), size=size, mode='strict')
            cutout2d_arr[i] = cutout2d.data / cutout2d.data.max()

            initial_guess = (1, 0, 0, aper_radius, 0)

            try:
                popt, _ = curve_fit(
                    CircularGaussian, xy, cutout2d_arr[i].flatten(), p0=initial_guess)

                a, x0, y0, sigma, b = popt

                if (a > 0) & (xy_min < x0 < xy_max) & (xy_min < y0 < xy_max):
                    position_arr[i] = x0 + size // 2, y0 + size // 2
                    fwhm_arr[i] = sigma * gaussian_sigma_to_fwhm
                    cutout2d_arr[i] -= b

                else:
                    raise RuntimeError('No feature fitted in the given region.')

            except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
                pass

    # Output
    fwhm, fwhm_err = np.nanmean(fwhm_arr), np.nanstd(fwhm_arr, ddof=1)

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'FWHM':
        title = [f'{title} FWHM (source detection)', 
                 f'{title} FWHM (Gaussian2D fitting)']
    else:
        title = [f'FWHM (source detection)', f'FWHM (Gaussian2D fitting)']

    fig_path = _validatePath(save, path, title)

    if show | save:

        # Plot source detection
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        # Image
        extent = (
            0.5, data_arr_bkgsb.shape[1] + 0.5, 0.5, data_arr_bkgsb.shape[0] + 0.5
        )
        _plot2d(ax=ax, ccd=data_arr_bkgsb, cmap='Greys_r', extent=extent)
        (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        # Mask
        ax.imshow(
            mask_arr, cmap='Greys', alpha=0.3 * mask_arr.astype(int), origin='lower', 
            extent=extent)
        # Aperture
        apertures = RectangularAperture(
            positions=np.transpose([x + 1, y + 1]), w=size, h=size, theta=0)
        apertures.plot(ax, color='r', lw=1)
        # Setting
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title[0], fontsize=16)

        if save: plt.savefig(fig_path[0], dpi=100)

        if show: plt.show()

        plt.close()

        if (n_peak % 4) == 0:
            nrow = n_peak // 4
        else:
            nrow = n_peak // 4 + 1

        fig = plt.figure(figsize=(10, 2 * nrow), dpi=100)
        extent = (0.5, size + 0.5, 0.5, size + 0.5)
        for i in range(n_peak):
            ax = fig.add_subplot(nrow, 4, i + 1)
            # Image
            _plot2d(
                ax=ax, ccd=cutout2d_arr[i], cmap='Greys_r', extent=extent, cbar=False, 
                xlabel=None, ylabel=None)
            # Aperture
            apertures = CircularAperture(
                (position_arr[i][0] + 1, position_arr[i][1] + 1), r=(fwhm_arr[i] / 2))
            apertures.plot(ax, color='r', lw=1)
            # Centroid
            ax.plot(position_arr[i][0] + 1, position_arr[i][1] + 1, 'r+')
            # Settings
            ax.tick_params(
                which='major', direction='in', top=True, right=True, color='w', length=5, 
                width=1, labelsize=12)
            ax.annotate(
                'FWHM$=' + f'{fwhm_arr[i]:.2f}' + '\\,$px', xy=(0.05, 0.85), 
                xycoords='axes fraction', color='w', fontsize=12)
        fig.supxlabel('column', fontsize=16)
        fig.supylabel('row', fontsize=16)
        fig.suptitle(title[1], fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path[1], dpi=100)

        if show: plt.show()

        plt.close()
    
    return fwhm, fwhm_err
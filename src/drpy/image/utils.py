import warnings

# NumPy
import numpy as np
# SciPy
from scipy.optimize import curve_fit, OptimizeWarning
# matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# AstroPy
from astropy.nddata import Cutout2D
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clip
# photutils
from photutils import CircularAperture, RectangularAperture
from photutils.background import Background2D
from photutils.centroids import centroid_com
from photutils.detection import find_peaks
# drpy
from drpy import conf
from drpy.modeling.function import CircularGaussian
from drpy.plotting import _plot2d
from drpy.validate import (_validateString, _validateBool, _validateRange, 
                           _validateInteger, _validateCCD, _validatePath)

# Set plot parameters
plt.rcParams['figure.figsize'] = [conf.fig_width, conf.fig_width]
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['getFWHM']


def getFWHM(ccd, box_size, n_sigma, aper_radius, saturation, n_peak, n_iter=5, 
            sigma_lower=None, sigma_upper=None, use_mask=False, title='fwhm', 
            show=conf.fig_show, save=conf.fig_save, path=conf.fig_path, **kwargs):
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

    n_iter : int, optional
        Number of sigma slipping iterations. Must be >= `0`.
        Default is `5`.

    sigma_lower : scalar or `None`, optional
        Number of standard deviations to use as the lower bound for the clipping limit. 
        If `None` (default), `3` is used.

    sigma_upper : scalar or `None`, optional
        Number of standard deviations to use as the upper bound for the clipping limit. 
        If `None` (default), `3` is used.

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
                    fwhm_arr[i] = np.abs(sigma) * gaussian_sigma_to_fwhm
                    cutout2d_arr[i] -= b

                else:
                    raise RuntimeError('No feature fitted in the given region.')

            except (RuntimeError, TypeError, OptimizeWarning): # raise exception here
                pass

    # Output
    fwhm_arr_masked, threshold_lower, threshold_upper = sigma_clip(
        data=fwhm_arr, sigma_lower=sigma_lower, sigma_upper=sigma_upper, 
        maxiters=n_iter, stdfunc='mad_std', masked=True, return_bounds=True)
    fwhm, fwhm_err = fwhm_arr_masked.mean(), fwhm_arr_masked.std(ddof=1)

    if fwhm_arr_masked.mask.sum() > 0:
        warnings.warn(
            'Fitting for some sources failed (may caused by cosmic rays).', 
            RuntimeWarning
        )

    # Plot
    _validateBool(show, 'show')
    _validateBool(save, 'save')

    if show | save:

        _validateString(title, 'title')
        if title != 'fwhm':
            title = [f'{title} fwhm (source detection)', 
                     f'{title} fwhm (cutoff)']
        else:
            title = [f'fwhm (source detection)', f'fwhm (cutoff)']

        # Source detection plot
        fig, ax = plt.subplots(1, 1, dpi=100)

        # Image
        _plot2d(ax=ax, ccd=data_arr_bkgsb, cmap='Greys_r')
        (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
        # Mask
        ax.imshow(
            mask_arr, cmap='Greys', alpha=0.3 * mask_arr.astype(int), origin='lower')
        # Aperture
        apertures = RectangularAperture(
            positions=np.transpose([x, y]), w=size, h=size, theta=0)
        apertures[fwhm_arr_masked.mask].plot(ax, color='b', lw=1)
        apertures[~fwhm_arr_masked.mask].plot(ax, color='r', lw=1)

        # Setting
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.annotate(
            'masked', xy=(0.75, 0.88), xycoords='axes fraction', fontsize=16, 
            color='b')
        ax.annotate(
            'detected', xy=(0.75, 0.93), xycoords='axes fraction', fontsize=16, 
            color='r')
        ax.set_title(title[0], fontsize=16)
        fig.set_figheight(
            (data_arr_bkgsb.shape[0] / data_arr_bkgsb.shape[1] * fig.get_figwidth()))
        fig.tight_layout()

        if save:
            fig_path = _validatePath(path, title[0])
            plt.savefig(fig_path, dpi=100)

        if show:
            plt.show()

        plt.close()

    # Cutout
    if save:

        fig_path = _validatePath(path, title[1], '.pdf')
        with PdfPages(fig_path, keep_empty=False) as pdf:

            for i in range(n_peak):

                if fwhm_arr_masked.mask[i]:
                    color = 'b'
                else:
                    color = 'r'

                fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=100)

                # Image
                _plot2d(
                    ax=ax, ccd=cutout2d_arr[i], cmap='Greys_r', cbar=False, 
                    xlabel='column', ylabel='row')
                # Aperture
                apertures = CircularAperture(
                    (position_arr[i][0], position_arr[i][1]), r=(fwhm_arr[i] / 2))
                apertures.plot(ax, color=color, lw=1)
                # Centroid
                ax.plot(position_arr[i][0], position_arr[i][1], '+', color=color, ms=15)
                
                # Settings
                ax.tick_params(
                    which='major', direction='in', top=True, right=True, color='w', 
                    length=5, width=1, labelsize=12)
                ax.annotate(
                    'FWHM$=' + f'{fwhm_arr[i]:.2f}' + '\\,$px', xy=(0.05, 0.85), 
                    xycoords='axes fraction', color='w', fontsize=12)
                ax.set_title(f'cutoff at ({x[i]:.0f}, {y[i]:.0f})', fontsize=16)
                fig.tight_layout()

                pdf.savefig(fig, dpi=100)

                plt.close()
    
    return fwhm, fwhm_err
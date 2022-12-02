from copy import deepcopy
import os

# NumPy
import numpy as np
# matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
# AstroPy
from astropy.visualization import ZScaleInterval
# drpsy
from drpsy import conf

from .validate import (_validateBool, _validatePath, _validateRange, _validateString, 
                       _validateSpectrum)

# Set plot parameters
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

__all__ = ['plotFitting', 'plotSpectrum1D', 'plot2d']


# todo: add _plotFitting?
def plotFitting(x, y, residual, mask, y_fit, x_fit=None, threshold_lower=None, 
                threshold_upper=None, xlabel='x', ylabel='y', title='fitting', 
                show=conf.show, save=conf.save, path=conf.path, use_relative=False):
    """Plot fitting.

    Parameters
    ----------
    show : bool, optional
        Whether to show plot.

    save : bool, optional
        Whether to save plot.

    path : bool, optional
        Path to save plot in.
    """

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'fitting':
        title = f'{title} fitting'

    fig_path = _validatePath(save, path, title)

    if show | save:
        
        if x_fit is None:
            x_fit = deepcopy(x)

        if threshold_lower is not None:
            if np.isnan(threshold_lower) | np.isinf(threshold_lower):
                threshold_lower = None
            else:
                _validateRange(
                    threshold_lower, 'threshold_lower', (None, 0), (None, True))

        if threshold_upper is not None:
            if np.isnan(threshold_upper) | np.isinf(threshold_upper):
                threshold_upper = None
            else:
                _validateRange(
                    threshold_upper, 'threshold_upper', (0, None), (True, None))

        _validateString(xlabel, 'xlabel')
        _validateString(ylabel, 'ylabel')

        _validateBool(use_relative, 'use_relative')
        if use_relative:
            label_residual = 'rel. residual'
        else:
            label_residual = 'residual'

        fig = plt.figure(figsize=(6, 6), dpi=100)
        gs = gridspec.GridSpec(3, 1)
        ax = fig.add_subplot(gs[:2]), fig.add_subplot(gs[2])

        # Data
        ax[0].plot(x[mask], y[mask], '+', c='lightgrey', ms=8)
        ax[0].plot(x[~mask], y[~mask], '+', c='black', ms=8)
        # Fitted data
        ax[0].plot(x_fit, y_fit, '-', c='red', lw=1.5)
        # Settings
        ax[0].grid(True, ls='--')
        ax[0].set_xlim(x_fit[0], x_fit[-1])
        # ax[0].set_ylim(y_fit.min(), y_fit.max())
        ax[0].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[0].minorticks_off()
        ax[0].set_xticklabels([])
        ax[0].set_ylabel(ylabel, fontsize=16)

        # Residual
        ax[1].plot(x[mask], residual[mask], '+', c='lightgrey', ms=8)
        ax[1].plot(x[~mask], residual[~mask], '+', c='black', ms=8)
        # Settings
        ax[1].grid(True, ls='--')
        ax[1].axhline(y=0, ls='-', c='red', lw=1.5)
        if threshold_lower is not None:
            ax[1].axhline(y=threshold_lower, ls='--', c='red', lw=1.5)
        if threshold_upper is not None:
            ax[1].axhline(y=threshold_upper, ls='--', c='red', lw=1.5)
        ax[1].set_xlim(x_fit[0], x_fit[-1])
        if threshold_lower is not None:
            ymin = 1.67 * threshold_lower
        else:
            ymin = None
        if threshold_upper is not None:
            ymax = 1.67 * threshold_upper
        else:
            ymax = None
        ax[1].set_ylim(ymin, ymax)
        ax[1].tick_params(
            which='major', direction='in', top=True, right=True, length=5, width=1.5, 
            labelsize=12)
        ax[1].minorticks_off()
        ax[1].set_xlabel(xlabel, fontsize=16)
        ax[1].set_ylabel(label_residual, fontsize=16)
        fig.align_ylabels()
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()


def _plotSpectrum1D(ax, spectral_axis, flux, uncertainty=None, xlabel='spectral axis', 
                    ylabel='flux'):
    """Plot 1-dimensional spectrum of type `~specutils.Spectrum1D`."""

    # 1-dimension
    if flux.ndim == 1:
        ax.step(spectral_axis, flux, where='mid', color='C0', lw=1.5, zorder=2.5)
        if uncertainty is not None:
            ax.fill_between(
                spectral_axis, flux - uncertainty, flux + uncertainty, color='C0', 
                alpha=0.2, zorder=2.5)
    # more than 1-dimension
    else:
        flux = flux.reshape(-1, flux.shape[-1])
        for i in range(flux.shape[0]):
            ax.step(spectral_axis, flux[i], where='mid', color='C0', lw=1.5, zorder=2.5)
            if uncertainty[i] is not None:
                ax.fill_between(
                    spectral_axis, (flux - uncertainty)[i], (flux + uncertainty)[i], 
                    color='C0', alpha=0.2, zorder=2.5)

    # Settings
    ax.grid(axis='both', color='0.95', zorder=-1)
    ax.set_xlim(spectral_axis[0], spectral_axis[-1])
    ax.tick_params(
        which='major', direction='in', top=True, right=True, length=5, width=1.5, 
        labelsize=12)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    
# todo: add mask, add legend?
def plotSpectrum1D(spectrum1d, title='spectrum', show=conf.show, save=conf.save, 
                   path=conf.path):
    """Plot 1-dimensional spectrum of type `~specutils.Spectrum1D`.
    
    Parameters
    ----------
    spectrum1d : `~specutils.Spectrum1D` or `~numpy.ndarray`
        Input spectrum.

    show : bool, optional
        Whether to show plot.

    save : bool, optional
        Whether to save plot.

    path : bool, optional
        Path to save plot in.
    """

    _validateBool(show, 'show')

    _validateString(title, 'title')
    if title != 'spectrum':
        title = f'{title} spectrum'

    fig_path = _validatePath(save, path, title)

    if show | save:

        new_spectrum1d, spectral_axis, flux, uncertainty, mask = _validateSpectrum(
            spectrum1d, 'spectrum1d', True, True)

        # Apply mask
        flux[mask] = np.nan
        uncertainty[mask] = np.nan

        # Labels
        unit_spectral_axis = new_spectrum1d.spectral_axis.unit.to_string()
        unit_flux = new_spectrum1d.flux.unit.to_string()

        xlabel = f'spectral axis [{unit_spectral_axis}]'
        ylabel = f'flux [{unit_flux}]'

        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        _plotSpectrum1D(ax, spectral_axis, flux, uncertainty, xlabel=xlabel, ylabel=ylabel)
        # ax.legend(fontsize=16)
        ax.set_title(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()


def _plot2d(ax, ccd, cmap='Greys_r', contrast=0.25, cbar=True, xlabel='column', 
            ylabel='row', cblabel='pixel value', **kwargs):
    """Plot image."""

    if ('vmin' not in kwargs) | ('vmax' not in kwargs):
        zscale = ZScaleInterval(
            nsamples=600, contrast=contrast, max_reject=0.5, min_npixels=5, krej=2.5, 
            max_iterations=5)
        if 'vmax' in kwargs:
            kwargs['vmin'], _ = zscale.get_limits(ccd)
        elif 'vmin' in kwargs:
            _, kwargs['vmax'] = zscale.get_limits(ccd)
        else:
            kwargs['vmin'], kwargs['vmax'] = zscale.get_limits(ccd)

    # Image
    im = ax.imshow(ccd, cmap=cmap, origin='lower', **kwargs)
    # Settings
    ax.tick_params(
        which='major', direction='in', top=True, right=True, color='w', length=5, 
        width=1.5, labelsize=12)
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    # Colorbar
    if cbar:
        cb = plt.colorbar(im, ax=ax, use_gridspec=True)
        # Settings
        cb.ax.tick_params(
            which='major', direction='in', right=True, color='w', length=5, width=1.5, 
            labelsize=12)
        cb.ax.set_ylabel(cblabel, fontsize=16)


def plot2d(ccd, cmap='Greys_r', contrast=0.25, extent=None, cbar=True, title='image', 
           show=conf.show, save=conf.save, path=conf.path, **kwargs):
    """Plot image.

    Parameters
    ----------
    show : bool, optional
        Whether to show plot.

    save : bool, optional
        Whether to save plot.

    path : bool, optional
        Path to save plot in.
    """

    _validateBool(show, 'show')

    _validateString(title, 'title')

    fig_path = _validatePath(save, path, title)

    if show | save:

        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        if extent is None:
            extent = (0.5, ccd.shape[1] + 0.5, 0.5, ccd.shape[0] + 0.5)
        _plot2d(
            ax=ax, ccd=ccd, cmap=cmap, contrast=contrast, cbar=cbar, extent=extent, 
            **kwargs)
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()

        if save: plt.savefig(fig_path, dpi=100)

        if show: plt.show()

        plt.close()
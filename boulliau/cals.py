from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.io import fits

__all__ = ['generate_master_flat_and_dark', 'generate_master_dark']


def generate_master_dark(dark_paths, master_dark_path):
    """
    Create a master flat from night-sky flats, and a master dark.

    Parameters
    ----------
    dark_paths : list
        List of paths to dark frames
    master_dark_path : str
        Path to master dark frame that will be created
    """
    # Make master dark frame:
    testdata = fits.getdata(dark_paths[0])
    allflatdarks = np.zeros((testdata.shape[0], testdata.shape[1],
                             len(dark_paths)))
    for i, darkpath in enumerate(dark_paths):
        allflatdarks[:, :, i] = fits.getdata(darkpath)
    masterflatdark = np.median(allflatdarks, axis=2)

    fits.writeto(master_dark_path, masterflatdark, clobber=True)


def generate_master_flat_and_dark(flat_paths, dark_paths, master_flat_path,
                                  master_dark_path):
    """
    Create a master flat from night-sky flats, and a master dark.

    Parameters
    ----------
    flat_paths : list
        List of paths to flat fields
    dark_paths : list
        List of paths to dark frames
    master_flat_path : str
        Path to master flat that will be created
    master_dark_path : str
        Path to master dark frame that will be created
    """

    # Make master dark frame:
    testdata = fits.getdata(dark_paths[0])
    dark_exposure_duration = fits.getheader(dark_paths[0])['EXPTIME']
    allflatdarks = np.zeros((testdata.shape[0], testdata.shape[1],
                             len(dark_paths)))
    for i, darkpath in enumerate(dark_paths):
        allflatdarks[:, :, i] = fits.getdata(darkpath)
    masterflatdark = np.median(allflatdarks, axis=2)

    fits.writeto(master_dark_path, masterflatdark, clobber=True)

    # Make master flat field:
    testdata = fits.getdata(flat_paths[0])
    flat_exposure_duration = fits.getheader(flat_paths[0])['EXPTIME']
    allflats = np.zeros((testdata.shape[0], testdata.shape[1], len(flat_paths)))

    for i, flatpath in enumerate(flat_paths):
        flat_dark_subtracted = (fits.getdata(flatpath) -
                                masterflatdark *
                                (flat_exposure_duration/dark_exposure_duration))
        allflats[:, :, i] = flat_dark_subtracted

    # do a median sky flat:

    coefficients = np.median(allflats, axis=2)
    coefficients[coefficients / np.median(coefficients) < 0.01] = np.median(coefficients)
    master_flat = coefficients/np.median(coefficients[coefficients != 1])
    fits.writeto(master_flat_path, master_flat, clobber=True)


def test_flat(image_path, master_flat_path, master_dark_path):
    import matplotlib.pyplot as plt

    image_no_flat = fits.getdata(image_path) - fits.getdata(master_dark_path)
    image = image_no_flat / fits.getdata(master_flat_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(image, origin='lower', interpolation='nearest',
                 cmap=plt.cm.viridis, vmin=np.percentile(image, 0.1),
                 vmax=np.percentile(image, 99.9))
    ax[1].hist(image_no_flat.ravel(), 200, label='No flat', alpha=0.4, log=True,
               histtype='stepfilled')
    ax[1].hist(image.ravel(), 200, label='Flat', alpha=0.4, log=True,
               histtype='stepfilled')
    ax[1].set_title(master_flat_path)
    ax[1].legend()

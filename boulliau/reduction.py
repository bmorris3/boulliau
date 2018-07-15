from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.utils.console import ProgressBar
from astropy.modeling import models, fitting
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import h5py
import matplotlib.pyplot as plt

from .photometry_results import PhotometryResults
from .star_selection import init_centroids

__all__ = ['photometry', 'force_photometry']


def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]//binning_factor, a.shape[1]//binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(sh).sum(-1).sum(1)


def photometry(image_paths, master_dark_path, master_flat_path, star_positions,
               aperture_radii, centroid_stamp_half_width, psf_stddev_init,
               aperture_annulus_radius):
    """
    Parameters
    ----------
    master_dark_path : str
        Path to master dark frame
    master_flat_path :str
        Path to master flat field
    target_centroid : `~numpy.ndarray`
        position of centroid, with shape (2, 1)
    comparison_flux_threshold : float
        Minimum fraction of the target star flux required to accept for a
        comparison star to be included
    aperture_radii : `~numpy.ndarray`
        Range of aperture radii to use
    centroid_stamp_half_width : int
        Centroiding is done within image stamps centered on the stars. This
        parameter sets the half-width of the image stamps.
    psf_stddev_init : float
        Initial guess for the width of the PSF stddev parameter, used for
        fitting 2D Gaussian kernels to the target star's PSF.
    aperture_annulus_radius : int
        For each aperture in ``aperture_radii``, measure the background in an
        annulus ``aperture_annulus_radius`` pixels bigger than the aperture
        radius
    """
    master_dark = fits.getdata(master_dark_path)
    master_flat = fits.getdata(master_flat_path)

    star_positions = np.array(star_positions)#.T

    # Initialize some empty arrays to fill with data:
    times = np.zeros(len(image_paths))
    fluxes = np.zeros((len(image_paths), len(star_positions),
                       len(aperture_radii)))
    errors = np.zeros((len(image_paths), len(star_positions),
                       len(aperture_radii)))
    xcentroids = np.zeros((len(image_paths), len(star_positions)))
    ycentroids = np.zeros((len(image_paths), len(star_positions)))
    airmass = np.zeros(len(image_paths))
    psf_stddev = np.zeros(len(image_paths))

    medians = np.zeros(len(image_paths))

    with ProgressBar(len(image_paths)) as bar:
        for i in range(len(image_paths)):
            bar.update()

            # Subtract image by the dark frame, normalize by flat field
            imagedata = (fits.getdata(image_paths[i]) - master_dark) / master_flat

            from scipy.ndimage import gaussian_filter

            smoothed_image = gaussian_filter(imagedata, 3)
            brightest_star_coords = np.unravel_index(np.argmax(smoothed_image),
                                                     smoothed_image.shape)

            if i == 0:
                brightest_start_coords_init = brightest_star_coords

            offset = np.array(brightest_start_coords_init) - np.array(brightest_star_coords)
            print('offset', offset)
            # Collect information from the header
            imageheader = fits.getheader(image_paths[i])
            exposure_duration = imageheader['EXPTIME']
            times[i] = Time(imageheader['DATE-OBS'], format='isot', scale=imageheader['TIMESYS'].lower()).jd
            medians[i] = np.median(imagedata)
            airmass[i] = imageheader['AIRMASS']

            # Initial guess for each stellar centroid informed by previous centroid
            for j in range(len(star_positions)):
                init_x = star_positions[j][0] + offset[0]
                init_y = star_positions[j][1] + offset[1]

                # Cut out a stamp of the full image centered on the star
                image_stamp = imagedata[int(init_y) - centroid_stamp_half_width:
                                        int(init_y) + centroid_stamp_half_width,
                                        int(init_x) - centroid_stamp_half_width:
                                        int(init_x) + centroid_stamp_half_width]

                x_stamp_centroid, y_stamp_centroid = np.unravel_index(np.argmax(image_stamp),
                                                                      image_stamp.shape)

                x_centroid = x_stamp_centroid + init_x - centroid_stamp_half_width
                y_centroid = y_stamp_centroid + init_y - centroid_stamp_half_width

                # plt.imshow(image_stamp, origin='lower')
                # plt.scatter(y_stamp_centroid, x_stamp_centroid)
                # plt.show()

                # plt.imshow(imagedata, origin='lower')
                # plt.scatter(x_centroid, y_centroid)
                #

                xcentroids[i, j] = x_centroid
                ycentroids[i, j] = y_centroid

                # For the target star, measure PSF:
                if j == 0:
                    psf_model_init = models.Gaussian2D(amplitude=np.max(image_stamp),
                                                       x_mean=centroid_stamp_half_width,
                                                       y_mean=centroid_stamp_half_width,
                                                       x_stddev=psf_stddev_init,
                                                       y_stddev=psf_stddev_init)

                    fit_p = fitting.LevMarLSQFitter()
                    y, x = np.mgrid[:image_stamp.shape[0], :image_stamp.shape[1]]
                    best_psf_model = fit_p(psf_model_init, x, y, image_stamp -
                                           np.median(image_stamp))
                    psf_stddev[i] = 0.5*(best_psf_model.x_stddev.value +
                                          best_psf_model.y_stddev.value)

            positions = np.vstack([ycentroids[i, :], xcentroids[i, :]]).T

            for k, aperture_radius in enumerate(aperture_radii):
                target_apertures = CircularAperture(positions, aperture_radius)
                background_annuli = CircularAnnulus(positions,
                                                    r_in=aperture_radius +
                                                         aperture_annulus_radius,
                                                    r_out=aperture_radius +
                                                          2 * aperture_annulus_radius)
                flux_in_annuli = aperture_photometry(imagedata,
                                                     background_annuli)['aperture_sum'].data
                background = flux_in_annuli/background_annuli.area()
                flux = aperture_photometry(imagedata,
                                           target_apertures)['aperture_sum'].data
                background_subtracted_flux = (flux - background *
                                              target_apertures.area())
                print(background, flux)
                # plt.imshow(smoothed_image, origin='lower')
                # target_apertures.plot()
                # background_annuli.plot()
                # plt.show()
                fluxes[i, :, k] = background_subtracted_flux/exposure_duration
                errors[i, :, k] = np.sqrt(flux)

    # Save some values
    results = PhotometryResults(times, fluxes, errors, xcentroids, ycentroids,
                                airmass, medians, psf_stddev, aperture_radii)
    return results


def force_photometry(image_paths, archive_path):
    """
    Perform forced photometry on a series of images.

    Will shift all images so that the brightest star is in the center, then
    do photometry on the next N brightest stars


    Parameters
    ----------
    image_paths : str
        String fed to `~glob.glob` to pick up FITS image paths.

    archive_path : str
        Path to an HDF5 archive of the images that will be created by this
        method

    Returns
    -------
    pr : `~boulliau.PhotometryResults`
        Results of the forced photometry
    """

    paths = sorted(glob(image_paths))

    image_shape = fits.getdata(paths[0]).shape

    f = h5py.File(archive_path, 'w')

    if 'images' not in f:
        dset = f.create_dataset("images", shape=(image_shape[0], image_shape[1],
                                                 len(paths)))
    else:
        dset = f['images']

    master_flat_path = 'tmp/masterflat.fits'
    master_dark_path = 'tmp/masterdark.fits'

    flat = fits.getdata(master_flat_path)
    dark = fits.getdata(master_dark_path)

    from skimage.feature import peak_local_max

    mid = image_shape[0]//2

    times = []
    airmass = []
    with ProgressBar(len(paths)) as bar:
        for i, path in enumerate(paths):

            raw_image = (fits.getdata(path) - dark) / flat
            times.append(fits.getheader(path)['JD'])
            airmass.append(fits.getheader(path)['AIRMASS'])

            coordinates = peak_local_max(raw_image, min_distance=5,
                                         num_peaks=1, exclude_border=0)
            y_mean = int(coordinates[:, 1].mean())
            x_mean = int(coordinates[:, 0].mean())

            firstroll = np.roll(raw_image, mid - y_mean,
                                axis=1)
            rolled_image = np.roll(firstroll, mid - x_mean,
                                   axis=0)

            dset[:, :, i] = rolled_image

            bar.update()

    dset.attrs['times'] = times
    dset.attrs['airmass'] = airmass

    # f.close()
    # f = h5py.File(archive_path, 'r')

    dset = f['images']
    background = np.median(dset[:], axis=(0, 1))

    times = dset.attrs['times']
    airmass = dset.attrs['airmass']

    centroids = init_centroids(dset[..., 0], [image_shape[0]//2, image_shape[1]//2],
                               min_flux=0.1)

    centroid_0 = centroids[:, 0].astype(int)
    centroid_1 = centroids[:, 1].astype(int)

    stamp_width = 10

    comparison1 = dset[centroid_0[0] - stamp_width:centroid_0[0] + stamp_width,
                       centroid_0[1] - stamp_width:centroid_0[1] + stamp_width, :]
    target = dset[centroid_1[0] - stamp_width:centroid_1[0] + stamp_width,
                  centroid_1[1] - stamp_width:centroid_1[1] + stamp_width, :]
    target_flux = np.sum(target, axis=(0, 1))
    comp_flux1 = np.sum(comparison1, axis=(0, 1))

    mask_outliers = np.ones_like(target_flux).astype(bool)

    X = np.vstack([comp_flux1, 1-airmass, background]).T

    c = np.linalg.lstsq(X[mask_outliers], target_flux[mask_outliers])[0]
    comparison = X @ c

    lc = target_flux/comparison

    pr = PhotometryResults(times=times, fluxes=lc)
    #pr.save(results_path)
    return pr
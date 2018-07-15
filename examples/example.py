import os
import h5py
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.utils.console import ProgressBar
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, '../')

from boulliau import PhotometryResults, init_centroids

np.random.seed(1984)

n_images = 20
period = 10
dims = (50, 50)


def generate_example_images():
    image = np.ones(dims) + 0.5 * np.random.randn(*dims)

    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')

    for i in range(n_images):
        image_i = image.copy()

        offset0 = np.random.randint(-3, 3)
        offset1 = np.random.randint(-3, 3)

        # Create target star with apparent rotation period
        star_width = 4
        image_i[dims[0]//2 + offset0 : dims[0]//2 + offset0 + star_width,
        dims[1]//2 + offset1 : dims[1]//2 + offset1 + star_width] = 20 * np.sin((2 * np.pi)/period * i) + 100

        # Create comparison star with constant flux
        image_i[10 + offset0 : 10 + offset0 + star_width, 10 + offset1 : 10 + offset1 + star_width] = 200

        # Create a FITS header
        hdu = fits.PrimaryHDU()
        hdu.header['JD'] = i
        hdu.header['EXPTIME'] = 1
        hdu.header['DATE-OBS'] = (Time('2018-07-15 00:00') + i*u.day).isot
        hdu.header['TIMESYS'] = 'tai'
        hdu.header['AIRMASS'] = 1
        fits.writeto('tmp/{0:02d}.fits'.format(i), image_i, overwrite=True, header=hdu.header)


def generate_masterdark_masterflat():
    fits.writeto('tmp/masterdark.fits', np.ones(dims), overwrite=True)
    fits.writeto('tmp/masterflat.fits', np.ones(dims), overwrite=True)


def create_archive():
    paths = sorted(glob('tmp/??.fits'))

    image_shape = fits.getdata(paths[0]).shape

    f = h5py.File('tmp/archive.hdf5', 'w')

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
    np.save('tmp/times.npy', times)
    np.save('tmp/airmass.npy', airmass)

    dset.attrs['times'] = times
    dset.attrs['airmass'] = airmass

    f.close()


def do_photometry():
    f = h5py.File('tmp/archive.hdf5', 'r')
    # times = np.loadtxt('tmp/times.npy')
    # airmass = np.loadtxt('tmp/airmass.npy')

    dset = f['images']
    background = np.median(dset[:], axis=(0, 1))

    times = dset.attrs['times']
    airmass = dset.attrs['airmass']
    #
    # plt.figure()
    # plt.imshow(dset[..., 0][:])
    # plt.show()

    centroids = init_centroids(dset[..., 0], [dims[0]//2, dims[1]//2],
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
    pr.save('lc.npz')

    plt.plot(times, lc)
    plt.show()

generate_example_images()
generate_masterdark_masterflat()
create_archive()
do_photometry()

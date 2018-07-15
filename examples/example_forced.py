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

from boulliau import PhotometryResults, force_photometry

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

generate_example_images()
generate_masterdark_masterflat()

pr = force_photometry("tmp/??.fits", 'tmp/archive.hdf5')

plt.plot(pr.times, pr.fluxes)
plt.show()


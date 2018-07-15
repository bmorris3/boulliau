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

from boulliau import photometry

np.random.seed(1984)

n_images = 20
period = 10
dims = (50, 50)
comparison_init_pos = [10, 10]
star_width = 4

def generate_example_images():
    image = np.ones(dims) + 0.5 * np.random.randn(*dims)

    if not os.path.exists('tmp/'):
        os.makedirs('tmp/')

    for i in range(n_images):
        image_i = image.copy()

        offset0 = np.random.randint(-1, 1)
        offset1 = np.random.randint(-1, 1)

        # Create target star with apparent rotation period

        image_i[dims[0]//2 + offset0 : dims[0]//2 + offset0 + star_width,
                dims[1]//2 + offset1 : dims[1]//2 + offset1 + star_width] = 20 * np.sin((2 * np.pi)/period * i) + 100

        # Create comparison star with constant flux
        image_i[comparison_init_pos[0] + offset0 : comparison_init_pos[0] + offset0 + star_width,
                comparison_init_pos[1] + offset1 : comparison_init_pos[1] + offset1 + star_width] = 200

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

master_dark_path = 'tmp/masterdark.fits'
master_flat_path = 'tmp/masterflat.fits'
star_positions = [[dims[0]//2, dims[1]//2],
                  comparison_init_pos]
aperture_radii = [8]
centroid_stamp_half_width = 10
psf_stddev_init = star_width/2
aperture_annulus_radius = 2

pr = photometry(glob("tmp/??.fits"), master_dark_path, master_flat_path,
                star_positions, aperture_radii, centroid_stamp_half_width,
                psf_stddev_init, aperture_annulus_radius)

plt.plot(pr.times, pr.fluxes[:, 0]/pr.fluxes[:, 1])
plt.show()


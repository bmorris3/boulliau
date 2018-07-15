from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from matplotlib import pyplot as plt
from astropy.stats import mad_std
from photutils import CircularAperture

from astropy.convolution import convolve_fft, Tophat2DKernel

__all__ = ['init_centroids']


def init_centroids(first_image, target_centroid,
                   min_flux=0.2, plots=False):

    tophat_kernel = Tophat2DKernel(5)

    try:
        from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
    except ImportError:
        from scipy.fftpack import fft2, ifft2

    convolution = convolve_fft(first_image, tophat_kernel, fftn=fft2, ifftn=ifft2)

    convolution -= np.median(convolution)

    mad = mad_std(convolution)

    convolution[convolution < -5*mad] = 0.0

    from skimage.filters import threshold_yen
    from skimage.measure import label, regionprops

    thresh = threshold_yen(convolution)

    masked = np.ones_like(convolution)
    masked[convolution <= thresh] = 0

    label_image = label(masked)

    # plt.figure()
    # plt.imshow(label_image, origin='lower', cmap=plt.cm.viridis)
    # plt.show()

    regions = regionprops(label_image, first_image)

    # reject regions near to edge of detector
    buffer_pixels = 5
    regions = [region for region in regions
               if ((region.weighted_centroid[0] > buffer_pixels and
                   region.weighted_centroid[0] < label_image.shape[0] - buffer_pixels)
               and (region.weighted_centroid[1] > buffer_pixels and
                    region.weighted_centroid[1] < label_image.shape[1] - buffer_pixels))]

    target_intensity = regions[0].mean_intensity

    centroids = [region.weighted_centroid for region in regions
                 if min_flux * target_intensity < region.mean_intensity]

    distances = [np.sqrt((target_centroid[0] - d[0])**2 +
                         (target_centroid[1] - d[1])**2) for d in centroids]

    centroids = np.array(centroids)[np.argsort(distances)]

    positions = np.vstack([[y for x, y in centroids], [x for x, y in centroids]])

    if plots:
        apertures = CircularAperture(positions, r=12.)
        apertures.plot(color='r', lw=2, alpha=1)
        plt.imshow(first_image, vmin=np.percentile(first_image, 0.01),
                   vmax=np.percentile(first_image, 99.9), cmap=plt.cm.viridis,
                   origin='lower')
        plt.scatter(positions[0, 0], positions[1, 0], s=150, marker='x')

        plt.show()
    return positions

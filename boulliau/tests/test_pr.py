
import numpy as np

from ..photometry_results import PhotometryResults


def test_photometryresults():
    n = 10
    times = np.ones(n)
    fluxes = np.ones(n)
    errors = np.ones(n)
    xcentroids = np.ones(n)
    ycentroids = np.ones(n)
    airmass = np.ones(n)
    background_median = np.ones(n)
    psf_stddev = np.ones(n)
    aperture_radii = np.ones(n)

    pr = PhotometryResults(times, fluxes, errors, xcentroids, ycentroids,
                           airmass, background_median, psf_stddev,
                           aperture_radii)

    pr.save('tmp.npz')

    pr_loaded = PhotometryResults.load('tmp.npz')

    np.testing.assert_allclose(pr_loaded.times, times)
    np.testing.assert_allclose(pr_loaded.fluxes, fluxes)
    np.testing.assert_allclose(pr_loaded.errors, errors)

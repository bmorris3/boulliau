import numpy as np

__all__ = ['PhotometryResults']


class PhotometryResults(object):
    def __init__(self, times, fluxes, errors=None, xcentroids=None,
                 ycentroids=None, airmass=None, background_median=None,
                 psf_stddev=None, aperture_radii=None):
        self.times = times
        self.fluxes = fluxes
        self.errors = errors
        self.airmass = airmass
        self.background_median = background_median
        self.psf_stddev = psf_stddev
        self.xcentroids = xcentroids
        self.ycentroids = ycentroids
        self.aperture_radii = aperture_radii

        self.attrs = ("times, fluxes, errors, airmass, "
                      "background_median, psf_stddev, xcentroids, "
                      "ycentroids, aperture_radii".split(', '))

    def save(self, path):
        """
        Save photometry results.

        Parameters
        ----------
        path : str
            Path to results to save.
        """
        np.savez(path, **{attr: getattr(self, attr) for attr in self.attrs})

    @classmethod
    def load(cls, path):
        """
        Load photometry results from an output file.

        Parameters
        ----------
        path : str
            Path to results to load.
        """
        load_file = np.load(path)
        return cls(**{key: load_file[key] for key in load_file})

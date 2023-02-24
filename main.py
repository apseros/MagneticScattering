import numpy as np

import structures
import plotting
from scatter import *

if __name__ == '__main__':
    # define magnetic configuration to create Sample class
    # options: vortex, skyrmion
    mag_config = structures.skyrmion(16, 16)
    mag_config = structures.tessellate(mag_config, times=[20, 20], pattern='hex')

    # create sample (sample dimensions [m], scattering factors [f0, f1, f2], magnetic configuration [q, mx, my, mz])
    sample1 = Sample(100e-6, [1 + 1j, 1 + 1j, 1j], mag_config)

    # define beam (wavelength [m], fwhm [m], polarization in stokes parameters[s0, s1, s2, s3])
    beam1 = Beam(17.6e-10, [60e-6, 60e-6], [1, 1, 0, 0])
    beam2 = Beam(17.6e-10, [60e-6, 60e-6], [1, -1, 0, 0])

    # experimental geometry (angle [deg], detector distance[m])
    exp1 = Geometry(0, 3)

    # calculate scattering for circular plus and circular minus
    scatter_lh = Scatter(beam1, sample1, exp1)
    scatter_lv = Scatter(beam2, sample1, exp1)

    # plot the magnetisation, resulting polarizations and scattering
    plotting.plot_structure(sample1, quiver=True)
    plotting.plot_intensity(scatter_lh, log=True)
    plotting.plot_intensity(scatter_lv, log=True)
    plotting.plot_diff(scatter_lh, scatter_lv, log=True)
    # ---------------------------------------ROTATE STRUCTURE---------------------------------------------
    # number of times to rotate the structure over a total of 180 degrees
    n_points = 18
    result = np.empty((n_points,) + mag_config[0].shape)

    # mask sample with circle to reduce rotation noise
    mag_config = structures.circle_mask(mag_config)
    sample1 = Sample(100e-6, [1 + 1j, 1 + 1j, 1j], mag_config)
    beam1 = Beam(17.6e-10, [30e-6, 30e-6], [1, 1, 0, 0])
    beam2 = Beam(17.6e-10, [30e-6, 30e-6], [1, -1, 0, 0])

    for i in range(n_points):
        sample1.rotate(180 * i / n_points, order=1)
        scatter_lh = Scatter(beam1, sample1, exp1)
        scatter_lv = Scatter(beam2, sample1, exp1)
        result[i] = np.log(1+scatter_lh.intensity) - np.log(1+scatter_lv.intensity)

    plotting.plot_3d(result.swapaxes(0, 2), extent=scatter_lh.extent)

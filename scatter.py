import numpy as np
import structures
import copy
from scipy.linalg import block_diag
from scipy.ndimage import affine_transform
from dataclasses import dataclass
from typing import Union

__all__ = ['Beam', 'Sample', 'Geometry', 'Scatter']


@dataclass
class Beam:
    """Describes the properties of the beam.

    Attributes:
        -wavelength :class:`float`:     The wavelength of the beam [m].
        -fwhm :class:`np.ndarray`:      The full width at half maximum of the beam in x and y.
        -pol :class:`list[float]`:      The polarization of the beam in Stokes parameters, 4 component list.
    """

    wavelength: float
    fwhm: Union[list[float], float, np.ndarray]
    pol: list[float]
    density_matrix: np.ndarray = None
    beam_profile: np.ndarray = None

    def __post_init__(self):
        """Calculates class attributes after initialization"""
        self.fwhm = cast_2d(self.fwhm)
        self.beam_sigma = self.fwhm / np.sqrt(8 * np.log(2))
        self.calculate_density_matrix()

    def calculate_density_matrix(self):
        """Converts the Stokes polarization vector to a density matrix

        The expression for converting the Strokes polarization "vector" to a density matrix is:
        rho = 1 / 2 * (P . sigma)
        where P is the four-dimensional vector and sigma is the vector containing the Pauli spin matrices and the
        identity matrix (sigma_0 = I).
        """
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = self.pol[0] + self.pol[1]
        rho[0, 1] = self.pol[2] - 1j * self.pol[3]
        rho[1, 0] = self.pol[2] + 1j * self.pol[3]
        rho[1, 1] = self.pol[0] - self.pol[1]
        self.density_matrix = 1 / 2 * rho

    def calculate_beam_profile(self, nx, ny, pix_size_x, pix_size_y) -> None:
        """Calculates the beam profile, which will be convolved with the structure.

        :param nx:          Number of pixels in x.
        :param ny:          Number of pixels in y.
        :param pix_size_x:  Pixel size in x.
        :param pix_size_y:  Pixel size in y.
        :return:            None.
        """
        x, y = structures.create_mesh(nx, ny)
        if np.any(self.fwhm == 0):
            self.beam_profile = np.ones((nx, ny))
        else:
            gaussian = gaussian_nd([x * pix_size_x, y * pix_size_y], self.fwhm)
            norm_gaussian = gaussian / np.max(gaussian)
            self.beam_profile = norm_gaussian


@dataclass
class Sample:
    """Describes the sample properties.

    Attributes:
        -sample_length :class:`np.ndarray`:          The dimensions of the sample [m].
        -scattering_factors :class:`list[complex]`:  List of the three complex scattering factors.
        -structure :class:`np.ndarray`:              The magnetic configuration of the sample, (4, X, Y) numpy array.
        -reference_structure :class:`np.ndarray`:    Backup of the original structure for when the structure is rotated.
    """
    sample_length: Union[list[float], float, np.ndarray]
    scattering_factors: list[complex]
    structure: np.ndarray = None
    reference_structure: np.ndarray = None

    def __post_init__(self):
        """Calculates class attributes after initialization.

        If no structure is given, a default skyrmion structure is used
        """
        if self.structure is None:
            tesselation_number = 20
            unit_cell_pixels = [16, 16]
            unit_cell = structures.skyrmion(*unit_cell_pixels)
            self.structure = structures.tessellate(unit_cell, tesselation_number, 'hex')
        elif self.structure.ndim != 3 or self.structure.shape[0] != 4:
            print("Structure has wrong shape, must be (4, x, y).")
            raise ValueError

        self.sample_length = cast_2d(self.sample_length)

        if len(self.scattering_factors) != 3:
            print("scattering_factors has wrong length, must be list of length 3.")
            raise ValueError

        nx, ny = self.structure.shape[1:3]
        self.pix_size_x, self.pix_size_y = np.divide(self.sample_length, [nx, ny])

    def get_pix_size(self) -> np.ndarray:
        return np.array([self.pix_size_x, self.pix_size_y])

    def rotate(self, angle: float, order: int = 1):
        """Rotates the structure in the xy plane.

        :param angle:   Angle of rotation [degrees].
        :param order:   Interpolation order, 1 - 5.
        :return:        None.
        """
        if self.reference_structure is None:
            self.reference_structure = copy.deepcopy(self.structure)
        else:
            self.structure = copy.deepcopy(self.reference_structure)
        angle = angle * np.pi / 180
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])

        self.structure = np.einsum('ij, jab -> iab', rot, self.structure[-3:])
        config_shape = np.asarray(self.structure.shape)
        rot = block_diag(1, rot[0:2, 0:2])
        out_center = rot @ (config_shape - 1) / 2
        in_center = (config_shape - 1) / 2
        offset = in_center - out_center
        if 0 <= order <= 5:
            self.structure = affine_transform(self.structure, rot, offset=offset, order=order, mode='nearest')
        else:
            raise Exception("Interpolation order not recognised.")


@dataclass
class Geometry:
    """Defines the geometry of the experiment.

    Attributes:
        angle :class:`float`:               The angle of incidence (and scattering) of the beam [degrees].
        detector_distance :class:`float`:   The distance between the sample and the detector [m].
    """
    angle: float
    detector_distance: float


class Scatter:
    """Class for defining a scattering experiment.
    """
    intensity = None
    pol_out = None
    convolved_structure = None
    extent = None

    def __init__(self, beam: Beam, sample: Sample, geometry: Geometry):
        """

        :param beam:        The beam properties form the Beam class.
        :param sample:      The sample properties from the Sample class.
        :param geometry:    The geometry of the experiment from the Geometry class.
        """
        self.beam = beam
        self.sample = sample
        self.geometry = geometry
        self.run()
        self._get_extent()

    def _get_extent(self):
        wavenumber = 2 * np.pi / self.beam.wavelength
        fourier_pix_size = np.pi / self.sample.get_pix_size()
        angular_space = fourier_pix_size / wavenumber
        real_space = angular_space * self.geometry.detector_distance
        self.extent = [-real_space[0], real_space[0], -real_space[1], real_space[1]]

    def run(self):
        """Prepare coordinates and run scattering method."""
        self.beam.calculate_beam_profile(*self.sample.structure.shape[1:3], *self.sample.get_pix_size())

        self.convolved_structure = self.sample.structure * self.beam.beam_profile
        scattering_factor = self._scattering_factor(self.convolved_structure)
        fft_scattering = np.fft.fftshift(np.fft.fft2(scattering_factor), axes=(-2, -1))

        self._compute_scattering(fft_scattering)

    def _compute_scattering(self, sf_fft):
        """Calculates scattering in q coordinates for the given polarization vector and structure scattering factors.

        The intensity if given by Tr(F mu F.T*).

        :param sf_fft:      FFT of the scattering factors and structure.
        :return:            None.
        """
        pauli_0 = np.eye(2)
        pauli_1 = np.array([[1, 0], [0, -1]])
        pauli_2 = np.array([[0, 1], [1, 0]])
        pauli_3 = np.array([[0, -1j], [1j, 0]])

        pauli_vec = np.array([pauli_0, pauli_1, pauli_2, pauli_3])
        rho_prime = np.einsum('ikab,kl,jlab->ijab', sf_fft, self.beam.density_matrix, sf_fft.conjugate())
        intensity = np.einsum('iiab->ab', rho_prime)
        pol_out = np.einsum('ijk,kjab->iab', pauli_vec, rho_prime)
        self.intensity = abs(intensity)
        self.pol_out = pol_out

    def _scattering_factor(self, fft_struct) -> np.ndarray:
        """Return the scattering factor for the given incident beam and magnetic structure.

        :param fft_struct:  Fourier transform of the magnetic structure, shaped (4, nx, ny).
        :return:            Complex scattering factor shaped (2, 2, nx, ny).
        """
        f_0, f_1, f_2 = self.sample.scattering_factors
        theta = self.geometry.angle * np.pi / 180
        s = np.sin(theta)
        c = np.cos(theta)
        s2 = np.sin(2 * theta)
        c2 = np.cos(2 * theta)

        if fft_struct.shape[0] == 3:
            m = fft_struct
            q = np.zeros_like(fft_struct[0])
        elif fft_struct.shape[0] == 4:
            m = fft_struct[1:4]
            q = fft_struct[0]
        else:
            raise ValueError("Structure has the wrong shape")

        nx, ny = self.sample.structure.shape[1:3]

        f = np.zeros((2, 2, nx, ny), dtype=complex)
        f[0, 0, :, :] = f_0 * q + f_2 * m[0] ** 2
        f[0, 1, :, :] = -1j * f_1 * (m[2] * c - m[1] * s) + f_2 * m[0] * (m[1] * c + m[2] * s)
        f[1, 0, :, :] = -1j * f_1 * (m[1] * s - m[2] * c) + f_2 * m[0] * (m[1] * c - m[2] * s)
        f[1, 1, :, :] = f_0 * q * c2 - 1j * f_1 * m[0] * s2 + f_2 * ((m[1] * c) ** 2 - (m[2] * s) ** 2)
        return f


def gaussian_nd(coords: list[np.ndarray], sigma: Union[float, list[float]] = 2.,
                mu: Union[float, list[float]] = 0.) -> np.ndarray:
    """Creates a Gaussian in N-dimensions, where N is the length of the coordinates list given.

    Example input:
        x, y = np.meshgrid(x, y)
        gaussian_nd([x, y], [2, 4], 0)
    Would return a 2D Gaussian with a standard deviation 2 along x and 4 along y, centered around 0.
    When scalars are an input for the last two options, the value is used in all dimensions.

    :param coords:  List with N elements, each being a coordinate mesh for the gaussian.
    :param sigma:   Standard deviation, list or scalar.
    :param mu:      Offset, list or scalar
    :return:        ND Gaussian
    """
    n_coords = len(coords)
    if isinstance(sigma, (int, float)):
        sigma = [sigma] * n_coords
    if isinstance(mu, (int, float)):
        mu = [mu] * n_coords
    if len(mu) != len(sigma) != n_coords:
        raise ValueError("Number of inputs doesn't match")

    result = np.zeros_like(coords, dtype=float)
    for i, (x, s, m) in enumerate(zip(coords, sigma, mu)):
        result[i] = np.exp(-(((x - m) / s) ** 2) / 2) / (s * np.sqrt(np.pi * 2))
    return np.prod(result, axis=0)


def cast_2d(data: Union[list[float], float, np.ndarray]):
    """Used for converting lists and floats into the appropriate shaped (2,) numpy array.

    :param data:    Input of scalar or list with 2 elements.
    :return:        Numpy array shaped (2,).
    """
    if type(data) is np.ndarray and data.shape != (2,):
        raise ValueError

    if type(data) is float:
        data = np.array([data] * 2)
    elif type(data) is list:
        if len(data) == 2:
            data = np.array(data)
        elif len(data) == 1:
            data = np.array(data * 2)
        else:
            raise ValueError

    return data

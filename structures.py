import numpy as np
from typing import Union


def create_mesh(*args: [list[int]]) -> list[np.ndarray]:
    """Creates a mesh centered around zero with the dimensions provided.

    Example:
        [x, y] = create_mesh(11, 8)
        Returns a list of numpy arrays of a fleshed out meshgrid with dimensions (11, 8) with the x dimension ranging
        from -5 to 5 and y from -3.5 to 3.5, both in steps on 1.

    :param args:    Integer dimensions.
    :return:        List of mesh grids with the provided dimensions.
    """
    if any(not isinstance(v, int) for v in args) or min(args) < 1:
        raise ValueError("Only positive integer values can be converted to a mesh.")
    vectors = map(lambda x: np.arange(x) - (x - 1) / 2, args)
    return np.meshgrid(*vectors, indexing='ij')


def cart2pol(x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> np.ndarray:
    """Converts cartesian coordinates to polar coordinates.

    :param x:   x coordinate.
    :param y:   y coordinate.
    :return:    Array of polar coordinates with the first index running over r and theta.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.array([r, theta])


def vortex(nx: int, ny: int) -> np.ndarray:
    """Creates a vortex with dimensions (nx, ny) at the center of the array.

    :param nx:  x size
    :param ny:  y size
    :return:    Vortex state with dimensions (3, x, y), where the first index runs over components.
    """
    min_dim = min([nx, ny])
    xx, yy = create_mesh(nx, ny)
    rad, theta = cart2pol(xx, yy)

    mx = -np.cos(theta)
    my = np.sin(theta)
    mz = np.zeros_like(theta)

    mx = np.where(rad > (min_dim - 1) / 2, 0, mx)
    my = np.where(rad > (min_dim - 1) / 2, 0, my)

    return np.array([np.zeros_like(mx), mx, my, mz])


def skyrmion(nx: int, ny: int, number: int = 1, helicity: float = 0, polarity: int = 1,
             neel: bool = False) -> np.ndarray:
    """Creates a skyrmion state with dimensions (nx, ny) at the center of the array.

    :param nx:          x size.
    :param ny:          y size.
    :param number:      Skyrmion topological number.
    :param helicity:    Helicity offset.
    :param polarity:    mz magnetization at the center of the skyrmion.
    :param neel:        True for Neel type skyrmion, False (default) for Bloch type.
    :return:            Skyrmion state with dimensions (3, x, y), where the first index runs over components..
    """

    xx, yy = create_mesh(nx, ny)
    azimuth = np.arctan2(xx, -yy)
    rad = np.sqrt(xx ** 2 + yy ** 2)
    # normalize theta to equal pi at the edges
    theta = 2 * rad / (np.min([nx, ny]) - 1) * np.pi

    mx = np.sin(theta) * np.cos(number * (azimuth + helicity))
    my = np.sin(theta) * np.sin(number * (azimuth + helicity))
    mz = polarity * np.cos(theta)

    if neel:
        mx, my = my, -mx
        mz = mz * neel * number

    mx = np.where(theta > np.pi, 0, mx)
    my = np.where(theta > np.pi, 0, my)
    mz = np.where(theta > np.pi, -polarity, mz)

    return np.array([np.zeros_like(mx), mx, my, mz])


def tessellate(unit_cell: np.ndarray, times: Union[int, list[int]], pattern: str = 'sq') -> np.ndarray:
    """Create a crystal from the given unit cell.

    :param unit_cell:       Structure to be tessellated, shaped (..., nx, ny)
    :param times:           The amount of times to tessellate structure in both dimensions
    :param pattern:         'hex' for hexagonal or 'sq' for square tiling
    :return:                Crustal structure shaped (..., nx*times, ny*times)
    """

    unit_shape = unit_cell.shape[1:3]
    nx, ny = unit_shape
    if type(times) is int:
        times = np.array([times, times])
    elif type(times) is list and len(times) == 2:
        times = np.array(times)
    else:
        print('Times must be an integer or a list of two integers.')
        return unit_cell

    if pattern[0:2] == ('sq' or 're'):
        return np.tile(unit_cell, times)
    elif pattern[0:3] == 'hex':
        if nx != ny:
            print('Hexagonal tiling only works with square unit cells.')
        final_shape = unit_shape * times
        times += 2
        filled = np.tile(unit_cell, times)
        select = [i + nx for i in range(final_shape[1] - nx) if i % (2 * nx) in range(nx)]
        partial_structure = filled[:, :, select]
        temp = np.roll(partial_structure, nx // 2, axis=1)
        filled[:, :, select] = temp
        return filled[:, nx:-nx, ny:-ny]
    else:
        print('Tesselation pattern not recognised.')
        return unit_cell


def circle_mask(structure: np.ndarray) -> np.ndarray:
    """Masks the structure with a circle of radius equal to the smallest dimension of the structure.

    :param structure:   Structure to be masked.
    :return:            Masked structure.
    """
    xx, yy = create_mesh(*structure.shape[-2:])
    boundary_values = [v[0, 0] for v in structure]
    rad = np.min([xx.max(), yy.max()])
    circle = (xx ** 2 + yy ** 2 < rad ** 2)
    for j in range(structure.shape[0]):
        structure[j] = structure[j] * circle + (1 - circle) * boundary_values[j]
    return structure


def pad_circle(structure: np.ndarray, times: float = 2, repeat_mode: str = 'edge') -> np.ndarray:
    """Applies a circular mask and then pads the structure with the given repeat mode.

    :param structure:       Structure to be padded.
    :param times:           The amount of times to pad the structure.
    :param repeat_mode:     The mode to use for padding.
    :return:                Padded structure.
    """
    structure = circle_mask(structure)
    times = tuple([2 * (int(times * size),) for size in structure.shape[-2:]])
    if structure.ndim == 3:
        times = ((0, 0), *times)
        return np.pad(structure, times, mode=repeat_mode)
    elif structure.ndim == 2:
        return np.pad(structure, times, mode=repeat_mode)
    else:
        raise ValueError("Structure must be 2 or 3 dimensional.")

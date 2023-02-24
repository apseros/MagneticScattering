import matplotlib.pyplot as plt
import numpy as np
from scatter import *
from typing import Union, Optional
from matplotlib import colors
from matplotlib.widgets import Slider


def plot_structure(structure: Union[Sample, np.ndarray], quiver=False) -> None:
    """Plot the components of magnetisation structure.

    :param structure:   Sample class or numpy array of shape (3, nx, ny).
    :param quiver:      Boolean choice to plot quiver plot of mx and my.
    :return:            None.
    """
    if isinstance(structure, Sample):
        structure_array = structure.structure
        array_shape = (np.array(structure_array.shape)[-2:] - 1) / 2
        pixel_size = structure.get_pix_size()
        limits = array_shape * pixel_size
        extent = [-limits[0], limits[0], -limits[1], limits[1]]

    elif isinstance(structure, np.ndarray):
        if structure.ndim != 3:
            raise ValueError("Structure must have 3 dimensions")
        elif structure.shape[0] < 3:
            raise ValueError("Structure must have 3 components")
        structure_array = structure
        extent = None

    else:
        raise ValueError("Structure must be a Sample class or numpy array.")

    if quiver and structure_array[0].size < 128 ** 2:
        plt.figure()
        plt.quiver(structure_array[-3], structure_array[-2], scale_units='dots')

    plot_struct, ax_struct = plt.subplots(1, 3)
    plot_struct.suptitle("Structure")

    for i in reversed(range(1, 4)):
        ax_struct[-i].imshow(structure_array[-i].T, origin='lower', extent=extent)
        ax_struct[-i].set_aspect('equal')

    ax_struct[-3].set_title("$m_x$")
    ax_struct[-2].set_title("$m_y$")
    ax_struct[-1].set_title("$m_z$")

    if extent is None:
        plot_struct.supxlabel("x / pixels")
        plot_struct.supylabel("y / pixels")
    else:
        plot_struct.supxlabel("x / m")
        plot_struct.supylabel("y / m")

    plt.show()


def plot_pol(scatter: Scatter) -> None:
    """Plot the polarization states of the scattered light.

    :param scatter:     Scatter class.
    :return:            None.
    """
    pol_out = np.abs(scatter.pol_out)
    extent = 2 * [-np.pi, np.pi]

    pol_fig, ax = plt.subplots(1, 3)
    pol_fig.suptitle("Relative Polarization States")
    pol_fig.supxlabel("$q_x$ / pixel$^{-1}$")
    pol_fig.supylabel("$q_y$ / pixel$^{-1}$")

    for i in range(3):
        ax[i].imshow(pol_out[i].T, origin='lower', extent=extent, vmin=pol_out.min(), vmax=pol_out.max())
        ax[i].set_aspect('equal')

    ax[0].set_title("Horizontal")
    ax[1].set_title("Diagonal")
    ax[2].set_title("Circular")

    plt.show()


def plot_intensity(scatter: Scatter, log: bool = False):
    """Plot the intensity of the scattered light.

    :param scatter:     Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :return:            None.
    """
    extent = scatter.extent

    if log:
        norm = colors.LogNorm(vmin=scatter.intensity.min() + 1, vmax=scatter.intensity.max())
    else:
        norm = None

    plt.figure()
    colorscale = plt.imshow(scatter.intensity.T, origin='lower', extent=extent, norm=norm)
    plt.colorbar(colorscale)
    plt.axis('scaled')
    plt.title("Intensity")
    plt.xlabel("Detector Position $x_0$ / m")
    plt.ylabel("Detector Position $y_0$ / m")

    plt.show()


def plot_diff(scatter_a: Scatter, scatter_b: Scatter, log: bool = False) -> None:
    """Plot the difference between two scattering patterns.

    :param scatter_a:   Scatter class.
    :param scatter_b:   Scatter class.
    :param log:         Boolean choice to plot in log scale.
    :return:            None.
    """
    if scatter_a.extent != scatter_b.extent:
        raise ValueError("Diffraction geometries have different parameters.")

    extent = scatter_a.extent
    if log:
        diff = np.log(1 + scatter_a.intensity) - np.log(1 + scatter_b.intensity)
        norm = colors.SymLogNorm(1, vmin=diff.min() + 1, vmax=diff.max())
    else:
        diff = scatter_a.intensity - scatter_b.intensity
        norm = None

    plt.figure()
    colorscale = plt.imshow(diff.T, origin='lower', extent=extent, norm=norm)
    plt.colorbar(colorscale)
    plt.axis('scaled')
    plt.title("Intensity")
    plt.xlabel("Detector Position $x_0$ / m")
    plt.ylabel("Detector Position $y_0$ / m")

    plt.show()


def plot_3d(data: np.ndarray, fig: plt.Figure = None, extent: list[float] = None) -> \
        Optional[tuple[plt.Figure, plt.Slider, plt.Slider]]:
    """Plot a 3D array as a 2D image with sliders to change the third dimension.

    :param data:    3D array to be plotted.
    :param fig:     Figure to plot on.
    :param extent:  Extent of the image in the form [xmin, xmax, ymin, ymax].
    :return:        Figure, slider for the first dimension, slider for the second dimension.
    """
    if data.ndim != 3:
        print("Data is not three-dimensional and cannot be plotted.")
        return None

    if fig is None:
        fig = plt.figure()

    data = data.swapaxes(0, 1)
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    image = ax.imshow(data[:, :, 0], origin='lower', extent=extent)

    # adjust the main plot to make room for the sliders
    vmin, vmax = np.min(data), np.max(data)
    fig.colorbar(image, ax=ax)
    image.set_clim([vmin, vmax])

    # Make a horizontal slider to control the frequency.
    ax_slice = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    slice_slider = Slider(ax=ax_slice, label='slice', valmin=0, valmax=data.shape[2] - 1, valstep=1, valinit=0)

    # Make a vertically oriented slider to control clim
    ax_clim = fig.add_axes([0.1, 0.25, 0.03, 0.5])
    clim_slider = Slider(ax=ax_clim, label='color limit', valmin=vmin, valmax=vmax, valinit=vmax,
                         orientation="vertical")

    # The function to be called anytime a slider's value changes
    def slice_update(val):
        image.set_data(data[:, :, int(val)])
        fig.canvas.draw_idle()

    def clim_update(val):
        image.set_clim([vmin, val])
        fig.canvas.draw_idle()

    # register the update function with each slider
    slice_slider.on_changed(slice_update)
    clim_slider.on_changed(clim_update)

    plt.show()

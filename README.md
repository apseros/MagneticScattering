# Magnetic Diffraction

Simple code for computing the magnetic diffraction pattern of a magnetic structure. 

To use: Define the structure, beam and geometry of the experiment, then compute scattering by calling the `Scatter` 
class. The magnetic structure is defined using the `Sample` class, the incident beam is defined using the `Beam` class 
and the geometry is defined using the `Geometry` class.

## Sample

The `Sample` is initialized with the following parameters: 

```python
Sample(sample_length, scattering_factors, magnetic_configuration)
```

`sample_length`: real size of the sample in meters. Scalar if x, y dimensions are equal or vector otherwise.

`scattering_factors`: scattering factors list with [f0, f1, f2] representing charge, XMCD and XMLD respectively.

`structure`: the magnetic configuration for scattering ([see section below](#structure)).

## Beam

The `Beam` is initialized with the following parameters:

```python
Beam(wavelength, beam_fwhm, polarization)
```

`wavelength`: wavelength of incident radiation in meters.

`beam_fwhm`: size of beam or full width at half maximum of the beam. Can be a scalar or a 2 component vector.

`polarization`: 4 component polarization in the form of a Stokes vector ([see section below](#stokes-vectors)).

## Geometry

The `Geometry` is initialized with the following parameters:

```python
Geometry(angle, detector_distance)
```

`angle`: The angle between the beam and the sample plane in degrees.

`detector_distance`: The distance between the sample and the detector in meters.

## Scatter
To compute the scattering pattern call the `Scatter` class with the following parameters:
```python
Scatter(Sample, Beam, Geometry)
```
The intensity of the scattering can be obtained from `Scatter.intensity` or plotted directly using functions in the 
plotting.py folder, some examples are:

```python
plotting.plot_structure(Sample, quiver=True)        # plot the components magnetic structure
plotting.plot_intensity(Scatter, log=True)          # plot the intensity of the scattering
plotting.plot_diff(Scatter_a, Scatter_b, log=True)  # plot the difference between two scattering patterns
```

## Stokes Parameters

The Stokes parameters are four components that define the polarization state of light. For convenience they are combined
to form a vector; it is actually a pseudovector and does not have any physical interpretation. They are defined as
follows:

`S_0`: Intensity of the light.

`S_1`: Component of intensity of light that is linearly polarized. Positive values for horizontal polarization, negative
values for vertical polarization.

`S_2`: Component of intensity of light that is linearly polarized along the diagonals. Positive values for +45 degree
polarization, negative values for -45 degree polarization.

`S_3`: Component of intensity of light that is circularly polarized. Positive values for right-handed polarization,
negative values for left-handed polarization.

## Structure

The magnetic vector field is a two-dimensional field, with three components. It is therefore defined as a `numpy` array
of shape `(3, nx, ny)`. To create this array from its scalar components, `(mx, my, mz)` one can use:

```python
structure = np.array([mx, my, mz])
```

`mx`, `my`, `mz` must all be of the same size and should be two dimensional, of size `(nx, ny)`, same as the pixel_size.

Structures can be made or imported using the `structures` header. Some examples are:

`strucutres.vortex(nx, ny)` and `structures.skyrmion(nx, ny)`.

## Geometry

The geometry here is such that the beam travels in the positive z dimension when `angle_d = 0` and along the negative y
dimension when `angle_d = 90`.

The sample plane is the x-y plane, thus the `mx` and `my` components are in-plane and `mz` is out of plane. 

# Theory of Magnetic Diffraction

Theory from "Soft X-ray resonant magnetic scattering of magnetic nanostructures"
, https://doi.org/10.1016/j.crhy.2007.06.004
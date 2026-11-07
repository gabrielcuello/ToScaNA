"""
transmission_vs_angle.py - Transmission of the scattered particles as a
function of the scattering angle 2theta (0 to 180 degrees), for ANY sample
geometry (slab or cylinder, see sample_base.py).

It combines the other modules:
    beam.py          -> where each particle starts (x, y) in the beam
    slab.py /
    cylinder.py      -> the sample: interaction points and path lengths
    sample_base.py   -> `scattered_direction`: direction for a given angle 2theta

Definition of the "transmission"
--------------------------------
For one scattering angle, the transmission is the probability that a
particle which interacted inside the sample leaves it towards the detector
without being attenuated:  T(2theta) = mean( exp(-sum_i mu_i * L_i) ),
averaged over all particles that interacted. L_i is the path in layer i
from the interaction point to the exit of the sample along the scattered
direction (Beer-Lambert law).

* The attenuation of the incident beam is NOT included again here, because
  it is already contained in the interaction points (they are drawn with the
  Beer-Lambert law along the beam path).
* The same particles are used for every angle ("common random numbers"),
  which gives smooth curves with little statistical noise.
* The detector is in the horizontal plane (azimuth phi = 0 means towards +x,
  phi = 180 towards -x), as in a 2-axis instrument.

Usage
-----
    python transmission_vs_angle.py          # runs the example at the bottom

or, from your own code:
    angles, T, f_int = transmission_vs_angle(sample, 8, 8, n_particles=50000)
    plot_transmission(angles, T)

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.linspace(a, b, n)    : n equally spaced numbers from a to b.
np.asarray              : convert to a NumPy array.
np.empty(n)             : an array of n numbers not yet filled in (we fill them in a loop).
np.random.default_rng   : random number generator.
a.mean()                : average of the elements.
a.shape / len(a)        : size of an array.
"""

import numpy as np

from .beam import sample_beam_point
from .slab import create_slab_sample
from .sample_base import scattered_direction


# --------------------------------------------------------------------------
# Calculation
# --------------------------------------------------------------------------
def transmission_vs_angle(sample, beam_width, beam_height, H=None, V=None,
                          n_particles=20000, angles_deg=None, phi_deg=0.0,
                          rng=None):
    """
    Compute the transmission of scattered particles for a list of angles.

    Parameters
    ----------
    sample : LayeredSample
        Slab or cylindrical sample (see sample_factory.make_sample).
    beam_width, beam_height : float
        Size of the incident beam (mm), centred on the beam axis. Parts of
        the beam that miss the sample are simply counted as "no interaction".
    H, V : callable, optional
        Horizontal and vertical beam profiles (see beam.py). Default: uniform.
    n_particles : int
        Number of incident particles to simulate. More particles give a
        smoother, more accurate curve.
    angles_deg : array-like, optional
        Scattering angles 2theta in degrees. Default: 0 to 180 in steps of 1.
    phi_deg : float
        Azimuth of the detector around the beam (0 = towards +x).
    rng : int or numpy.random.Generator, optional
        Seed or generator for reproducible results.

    Returns
    -------
    angles_deg : numpy.ndarray
        The scattering angles (degrees).
    transmission : numpy.ndarray
        Transmission at each angle (between 0 and 1), for the particles that
        interacted (see the definition at the top of this file).
    interaction_fraction : float
        Fraction of the incident particles that interacted in the sample.
        Multiply `transmission` by it to get the fraction of the incident
        beam that is scattered AND leaves the sample towards each angle.
    """
    # Default list of angles: 0, 1, 2, ..., 180 degrees (181 equally spaced values).
    if angles_deg is None:
        angles_deg = np.linspace(0.0, 180.0, 181)
    angles_deg = np.asarray(angles_deg, dtype=float)

    # Accept a seed or a ready-made random generator.
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # --- Step 1: where each particle starts (x, y) in the beam ---------------
    # (beam.py). n >= 2 so that arrays, not single numbers, are returned.
    x, y = sample_beam_point(beam_width, beam_height, H, V,
                             n=max(n_particles, 2), rng=rng)

    # --- Step 2: first interaction points in the sample ----------------------
    # The sample knows its own geometry (slab orientation, cylinder radii...).
    # `points` holds only the particles that interacted (one row per particle,
    # in the sample frame); particles that miss the sample or cross it without
    # interacting are dropped. `interacted` is a True/False table over ALL the
    # incident particles.
    points, interacted = sample.sample_interaction_points(x, y, rng=rng)
    interaction_fraction = float(interacted.mean())   # mean of True/False = fraction of True
    if len(points) == 0:
        raise RuntimeError("No particle interacted: increase n_particles, "
                           "check the attenuation coefficients or check that "
                           "the beam hits the sample.")

    # --- Step 3: for each angle, path to the exit and attenuation ------------
    transmission = np.empty(len(angles_deg))
    for i, angle in enumerate(angles_deg):        # enumerate gives (position, value) pairs
        # Direction of the detector in the lab frame...
        # ...converted to the sample frame, where the sample geometry is simple.
        direction = sample.to_sample_frame(scattered_direction(angle, phi_deg))
        # Distance travelled in each layer from each interaction point to the
        # exit of the sample along that direction: array (N, n_layers).
        path = sample.outgoing_path_lengths(points, direction)
        # Beer-Lambert: exp(-sum(mu*L)) for every particle, then average.
        transmission[i] = sample.transmission(path).mean()

    return angles_deg, transmission, interaction_fraction


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_transmission(angles_deg, transmission, title=None, save_as=None,
                      show=True):
    """
    Plot the transmission as a function of the scattering angle.

    transmission : one array, or a dictionary {label: array} to compare
                   several curves (e.g. several sample orientations).
    save_as      : file name (e.g. "transmission.png") to save the figure.
    show         : open the plot window (set False to only save the file).
    """
    import matplotlib.pyplot as plt     # matplotlib: the plotting library

    # Accept one curve or several (a dictionary label -> values).
    curves = (transmission if isinstance(transmission, dict)
              else {None: transmission})

    fig, ax = plt.subplots(figsize=(7, 4.5))     # a figure with one set of axes
    for label, values in curves.items():
        ax.plot(angles_deg, values, lw=2, label=label)
    ax.set_xlabel("Scattering angle 2$\\theta$ (degrees)")
    ax.set_ylabel("Transmission")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(title or "Transmission of scattered particles vs angle")
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()

    if save_as:
        fig.savefig(save_as, dpi=150)
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Example:  python transmission_vs_angle.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Slab: 20 x 20 mm, three layers (name, thickness mm, mu 1/mm).
    sample = create_slab_sample(20, 20, [("Al", 2.0, 0.020),
                                    ("Fe", 1.0, 0.250),
                                    ("Cu", 0.5, 0.300)],
                           plane_angle_deg=90)
    print(sample.summary())

    # Optional beam profile: Gaussian horizontally (sigma = 1.5 mm).
    # Use H=None, V=None for a uniform beam.
    gaussian_x = lambda x: np.exp(-x**2 / (2 * 1.5**2))

    # Same calculation for several orientations of the sample.
    curves = {}
    for plane_angle in (90, 60, 45, 30):
        rotated = sample.with_plane_angle(plane_angle)
        angles, T, f_int = transmission_vs_angle(
            rotated, beam_width=8, beam_height=8, H=gaussian_x,
            n_particles=50000, rng=1)
        curves[f"beam/plane angle = {plane_angle}°"] = T
        print(f"plane angle {plane_angle:3d} deg: fraction interacting = {f_int:.3f}")

    plot_transmission(angles, curves, save_as="transmission_vs_angle.png")

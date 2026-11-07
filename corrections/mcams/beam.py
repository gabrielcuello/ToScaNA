"""
beam.py - Where do the particles of a rectangular incident beam start?

The beam is a rectangle in the (x, y) plane, perpendicular to the direction
of flight z. The intensity across the beam is given by two independent
profiles: H(x) (horizontal) and V(y) (vertical). A particle starts at a point
(x, y) chosen at random with probability proportional to H(x) * V(y). With no
profile given, the beam is uniform.

Conventions
-----------
* All distances are in millimetres (mm).
* The particles fly along +z, perpendicular to the (x, y) plane.
* The beam rectangle is centred on the origin of the (x, y) plane:
      x in [-width/2,  +width/2]   (horizontal)
      y in [-height/2, +height/2]  (vertical)
* H(x) and V(y) are *relative intensity profiles*. They do not need to be
  normalised (only their shape matters) but must be >= 0 and not zero
  everywhere. If they are not given, the beam is uniform.

Main entry point
----------------
    sample_beam_point(width, height, H=None, V=None, n=1, rng=None)

Example
-------
    >>> import numpy as np
    >>> gaussian = lambda x: np.exp(-x**2 / (2 * 1.5**2))   # sigma = 1.5 mm
    >>> x, y = sample_beam_point(10, 6, H=gaussian)          # one particle
    >>> xs, ys = sample_beam_point(10, 6, H=gaussian, n=1000)  # many particles

NumPy functions used in this file (short glossary)
--------------------------------------------------
np.random.default_rng(seed) : create a random number generator (the seed makes
                              the results reproducible).
rng.uniform(a, b, size=n)   : n random numbers uniformly distributed in [a, b).
np.linspace(a, b, n)        : n equally spaced numbers from a to b (both included).
np.asarray(x, dtype=float)  : convert to a NumPy array of decimal numbers.
np.any / np.all             : True if any / all elements of a boolean array are True.
np.isfinite(a)              : True where a is neither infinite nor NaN ("not a number").
np.diff(a)                  : differences between neighbours: [1,4,9] -> [3,5].
np.cumsum(a)                : running sum: [1,2,3] -> [1,3,6].
np.concatenate              : join arrays one after the other.
np.interp(x, xp, fp)        : linear interpolation: value at x of the curve
                              made of the points (xp, fp). Here it is used to
                              invert a cumulative distribution.
"""

import numpy as np


# --------------------------------------------------------------------------
# Helper: sample a 1D coordinate from an arbitrary (unnormalised) profile
# --------------------------------------------------------------------------
def _sample_1d(length, profile, n, rng, n_grid=2001):
    """
    Draw `n` random positions in [-length/2, +length/2] distributed
    according to `profile`.

    Method: "inverse transform sampling". The profile is evaluated on a
    fine grid, integrated to get the cumulative distribution (CDF), and
    uniform random numbers are mapped through the inverse CDF.
    Why does it work? The CDF F(x) is the probability that the position is
    smaller than x. If u is uniform in [0, 1], then x = F^-1(u) has exactly
    the distribution we want: where the profile is high, F grows fast, so
    many values of u are mapped to that region.

    Parameters
    ----------
    length : float
        Size of the interval (mm).
    profile : callable or None
        Function of position (mm) giving the relative intensity.
        None means uniform.
    n : int
        Number of positions to draw.
    rng : numpy.random.Generator
        Random number generator.
    n_grid : int
        Number of grid points used to describe the profile.

    Returns
    -------
    numpy.ndarray of shape (n,)
    """
    half = length / 2.0

    # Uniform case: no need for any numerical work.
    if profile is None:
        return rng.uniform(-half, half, size=n)

    # --- Step 1: evaluate the profile on a fine, regular grid ---------------
    grid = np.linspace(-half, half, n_grid)
    try:
        values = np.asarray(profile(grid), dtype=float)   # vectorised call
        if values.shape != grid.shape:                    # e.g. returns a scalar
            raise ValueError
    except Exception:
        # Fallback for functions that only accept a single number: call them
        # once per grid point.
        values = np.array([profile(g) for g in grid], dtype=float)

    # --- Step 2: sanity checks on the profile -----------------------------------
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Profile must be finite and non-negative everywhere.")
    if values.sum() == 0:
        raise ValueError("Profile is zero everywhere inside the beam.")

    # --- Step 3: cumulative distribution (trapezoidal rule) ---------------------
    # Area of each trapezoid between neighbouring grid points: average height
    # times width. np.cumsum adds them up from the left; we put a 0 in front
    # (nothing accumulated at the left edge) and divide by the final value so
    # that the CDF goes from 0 to 1.
    steps = 0.5 * (values[1:] + values[:-1]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(steps)))
    cdf /= cdf[-1]

    # --- Step 4: map uniform random numbers through the inverse CDF -------------
    # np.interp(u, cdf, grid) reads the position `grid` that corresponds to
    # each cumulative probability u: this is F^-1(u).
    u = rng.uniform(0.0, 1.0, size=n)
    return np.interp(u, cdf, grid)


# --------------------------------------------------------------------------
# Main function
# --------------------------------------------------------------------------
def sample_beam_point(width, height, H=None, V=None, n=1, rng=None):
    """
    Pick the (x, y) point(s) in the beam rectangle where particle(s) start.

    Parameters
    ----------
    width : float
        Horizontal beam size along x (mm).
    height : float
        Vertical beam size along y (mm).
    H : callable, optional
        Horizontal profile H(x), x in mm. Default: uniform.
    V : callable, optional
        Vertical profile V(y), y in mm. Default: uniform.
    n : int, optional
        Number of particles (default 1).
    rng : numpy.random.Generator or int, optional
        Random generator, or a seed for reproducible results.

    Returns
    -------
    (x, y) : tuple of floats if n == 1, otherwise tuple of two arrays
        Starting coordinates in mm. The particle then flies along +z.

    Notes
    -----
    The two profiles are treated as independent, so the 2D intensity is
    the product H(x) * V(y).
    """
    if width <= 0 or height <= 0:
        raise ValueError("Beam width and height must be positive (mm).")
    if n < 1:
        raise ValueError("n must be at least 1.")

    # Accept a seed, an existing Generator, or nothing (then the generator
    # is seeded from the operating system: different results every run).
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Because H and V are independent, x and y are drawn separately.
    x = _sample_1d(width, H, n, rng)
    y = _sample_1d(height, V, n, rng)

    if n == 1:
        return float(x[0]), float(y[0])
    return x, y


def start_position(width, height, H=None, V=None, z0=0.0, rng=None):
    """
    Convenience wrapper: one particle, returned as a 3D point (x, y, z0)
    together with its direction of flight (0, 0, 1), i.e. along +z.

    z0 is the z position (mm) of the beam plane; default 0.
    """
    x, y = sample_beam_point(width, height, H, V, n=1, rng=rng)
    return (x, y, z0), (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------
# Demo / quick test:  python beam.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt     # plotting library, only needed here

    W, Hgt = 10.0, 6.0                                  # beam size in mm

    # Example profiles (mm): Gaussian horizontally, uniform vertically.
    # `lambda x: ...` defines a small anonymous function of x.
    gaussian_x = lambda x: np.exp(-x**2 / (2 * 1.5**2))

    # Single particle
    print("One start point (x, y) in mm:", sample_beam_point(W, Hgt, H=gaussian_x))

    # Many particles, to visualise the beam
    xs, ys = sample_beam_point(W, Hgt, H=gaussian_x, n=20000, rng=42)

    plt.figure(figsize=(6, 4))
    # hist2d counts how many particles fall in each little rectangle (bin).
    plt.hist2d(xs, ys, bins=60, range=[[-W / 2, W / 2], [-Hgt / 2, Hgt / 2]])
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Sampled start points: Gaussian H(x), uniform V(y)")
    plt.colorbar(label="counts")
    plt.tight_layout()
    plt.show()

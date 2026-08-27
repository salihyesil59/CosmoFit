"""
eBOSS DR16 BAO measurements distributed as likelihood *surfaces*.

Everything else in this library's BAO family is a mean and a
covariance. Two of eBOSS DR16's tracers are not, and in both cases
the collaboration released a grid precisely because a Gaussian would
misrepresent the measurement:

* **ELG** at ``z_eff = 0.845``. The BAO feature is detected at only
  1.4 sigma, so the likelihood is asymmetric -- published as
  ``D_V/r_d = 18.33 (+0.57/-0.62)`` -- and, more importantly, it has
  not decayed by the low edge of the released table. Roughly 11% of
  the probability sits below ``D_V/r_d = 16.5``, where a Gaussian
  centred on 18.33 with sigma 0.6 puts one part in a thousand. A
  Gaussian does not merely lose the asymmetry here; it loses two
  orders of magnitude of tail.

* **Lyman-alpha** at ``z_eff = 2.334``, a two-dimensional surface in
  ``(D_M/r_d, D_H/r_d)``. This one is closer to Gaussian than ELG --
  measured against a Gaussian with the same mean and covariance it
  departs by ``delta chi2 = 1.7`` out at 3 sigma, and the
  cross-correlation half alone by 5.0. What the table mainly buys
  here is the correlation itself, ``-0.46`` between the two ratios,
  which no pair of separate error bars carries.

The second is the one worth having. At ``z = 2.334`` this is the
highest-redshift BAO measurement in the library outside the CMB, and
it is independent of DESI's -- which matters directly for
``examples/lscdm_mcmc.ipynb``, where the constraint on the LsCDM
sign-switch redshift turned on a profile-likelihood cliff straddling
DESI's own Lyman-alpha bin at ``z = 2.33``.

Conventions
-----------
``chi2()`` returns ``-2 log L`` read from the table, following
:class:`~likelihoods.planck_lowe.PlanckLowEELikelihood`. It is not a
sum of squared pulls. The released grids are normalized to a maximum
probability of 1, so for a single-component version this is already
zero at the best-fitting distance ratio; the combined Lyman-alpha
version carries a small constant offset instead, because two
surfaces peaking in different places cannot both reach 1 at the same
point. The offset is the same for every cosmology and cancels in any
difference of chi2.

Outside the grid the likelihood is zero and this returns ``+inf``.
That is a statement about the data, not a numerical guard -- but for
ELG it is also a real prior, because the table's low edge is reached
at ``delta chi2 = 3.3``, still inside 2 sigma. The same truncation on
the Lyman-alpha grids costs nothing; there the surface has long
decayed.

References
----------
de Mattia et al. (2020), MNRAS 501, 5616,
`arXiv:2007.09008 <https://arxiv.org/abs/2007.09008>`_ (ELG).
du Mas des Bourboux et al. (2020), ApJ 901, 153,
`arXiv:2007.08995 <https://arxiv.org/abs/2007.08995>`_ (Lyman-alpha).
Grids as distributed by ``CobayaSampler/bao_data``.
"""

from __future__ import annotations

import numpy as np

from scipy.interpolate import (
    CubicSpline,
    RectBivariateSpline,
    RegularGridInterpolator,
)

from CosmoFit.data.loader import load_eboss_table

from .base import BaseLikelihood
from .desi import MODEL_MAP


class TabulatedBAOLikelihood(BaseLikelihood):
    """
    A BAO likelihood evaluated by interpolating a released grid.

    Subclassed rather than used directly:
    :class:`EBOSSELGLikelihood` and :class:`EBOSSLyaLikelihood` fix
    ``FAMILY`` and the display name.

    Parameters
    ----------
    cosmology : Cosmology
        Model to evaluate.
    version : str, optional
        Dataset version, passed to
        :func:`~data.loader.load_eboss_table`.
    """

    #: Loader family, set by the subclass.
    FAMILY: str = ""

    #: Display name, set by the subclass.
    LABEL: str = ""

    def __init__(
        self,
        cosmology,
        version: str = "dr16",
    ):

        dataset = load_eboss_table(

            self.FAMILY,

            version,

        )

        super().__init__(

            name=self.LABEL if version == "dr16"
                 else f"{self.LABEL} ({version})",

            dataset=dataset,

            cosmology=cosmology,

        )

        self.version = version

        self._interpolator = self._build_interpolator()

    # ---------------------------------------------------------

    def _build_interpolator(self):
        """
        Cubic spline through the *log* of the released probability.

        Splining the probability itself fails twice over: it spans
        thirty orders of magnitude, so the fit is dominated by the
        peak and ragged in the tails, and a cubic through
        near-zero nodes undershoots to negative values that have no
        logarithm. The log is smooth and near-quadratic around the
        peak -- which is what makes the recovered error bars match
        the published ones.
        """

        data = self.data

        if len(data.axes) == 1:

            return CubicSpline(

                data.axes[0],

                data.log_prob,

                extrapolate=False,

            )

        if len(data.axes) == 2:

            return RectBivariateSpline(

                data.axes[0],

                data.axes[1],

                data.log_prob,

                kx=3,

                ky=3,

            )

        # Three or more axes, and **linear**, which is the one place
        # this module does not spline.
        #
        # The 3-D grid is shipped with its log floored 200 below the
        # peak, because a tenth of the released probabilities
        # underflowed to exact zero. That floor is a plateau, and a
        # cubic through the step at its edge rings: measured on a
        # 120^3 mesh, the interpolated log reached **+146** against a
        # node maximum of 0. exp(146) is 1e63, so half a percent of
        # the volume would have swamped every normalization and put
        # the marginals at the grid edges -- which is exactly what
        # the published-value test caught.
        #
        # Linear interpolation is bounded by the surrounding nodes
        # and cannot overshoot. The cost is a piecewise-linear
        # surface with kinks at the nodes, which is acceptable at
        # 100 points per axis and is the right trade against being
        # wrong by sixty orders of magnitude.
        return RegularGridInterpolator(

            data.axes,

            data.log_prob,

            method="linear",

            bounds_error=False,

            fill_value=None,

        )

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted distance ratios, one per tabulated observable.
        """

        return np.array(

            [

                MODEL_MAP[observable](

                    self.cosmology,

                    self.data.z_eff,

                )

                for observable in self.data.observable

            ],

            dtype=float,

        )

    # ---------------------------------------------------------

    def log_likelihood_at(
        self,
        values,
    ):
        """
        Interpolate the released surface at given distance ratios,
        with no cosmology involved.

        Parameters
        ----------
        values : array_like
            One value per tabulated observable -- ``(D_V/r_d,)`` for
            ELG, ``(D_M/r_d, D_H/r_d)`` for Lyman-alpha. Leading
            axes are broadcast, so passing arrays evaluates the
            whole surface at once.

        Returns
        -------
        float or ndarray
            ``log L``, and ``-inf`` outside the tabulated range.

        Notes
        -----
        Public because the surface is worth looking at
        independently of any fit: this is what draws the likelihood
        contours, and it is what the tests use to check the grid
        against the published constraints without a cosmology
        standing in the way.
        """

        values = np.asarray(values, dtype=float)

        coordinates = [values[..., i] for i in range(len(self.data.axes))]

        inside = np.ones(coordinates[0].shape, dtype=bool)

        for value, (low, high) in zip(coordinates, self.data.bounds):

            inside &= (value >= low) & (value <= high)

        # Clip before interpolating: RectBivariateSpline extrapolates
        # a cubic outside its knots, which for a decaying surface
        # runs away rather than to zero. The clipped values are
        # masked out immediately below -- this only keeps the spline
        # from being asked a question with a nonsense answer.
        clipped = [
            np.clip(value, low, high)
            for value, (low, high) in zip(coordinates, self.data.bounds)
        ]

        if len(self.data.axes) == 1:

            result = self._interpolator(clipped[0])

        elif len(self.data.axes) == 2:

            result = self._interpolator(

                clipped[0],

                clipped[1],

                grid=False,

            )

        else:

            # RegularGridInterpolator wants (n_points, ndim) and
            # gives back (n_points,), so the caller's own shape --
            # scalar, vector, or a whole meshgrid -- is restored
            # afterwards rather than left as whatever it returned.
            points = np.stack(

                [value.ravel() for value in clipped],

                axis=-1,

            )

            result = self._interpolator(points).reshape(inside.shape)

        result = np.where(inside, result, -np.inf)

        return result if result.ndim else float(result)

    # ---------------------------------------------------------

    def log_likelihood(
        self,
    ) -> float:
        """
        Log-likelihood at the current cosmology.

        A prediction outside the tabulated range returns ``-inf``:
        the released surface is zero there, and a cosmology the data
        excludes is not something to extrapolate towards.
        """

        return float(self.log_likelihood_at(self.model()))

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Prediction minus the grid's most probable point.

        A diagnostic for plotting. The likelihood is not built from
        it -- the surface is not symmetric about its peak, which is
        the reason this dataset is a table.
        """

        return self.model() - np.asarray(self.data.peak, dtype=float)

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        ``-2 log L``, so this composes with the Gaussian
        likelihoods it is summed with.

        See the module docstring for what the zero point is.
        """

        log_like = self.log_likelihood()

        if not np.isfinite(log_like):

            return np.inf

        return -2.0 * log_like


class EBOSSELGLikelihood(TabulatedBAOLikelihood):
    """
    eBOSS DR16 emission-line galaxies, ``D_V/r_d`` at
    ``z_eff = 0.845``, as a tabulated likelihood.

    Warning
    -------
    Overlaps ``"desi"`` -- DESI's ELG sample succeeds this one over
    much of the same footprint. It does *not* conflict with
    ``"sdss_bao"``: eBOSS's own DR16 consensus combines the ELG
    sample with the LRG and QSO ones as independent tracers, and so
    does every reference implementation.
    """

    FAMILY = "eboss_elg"

    LABEL = "eBOSS-DR16-ELG"


class EBOSSLyaLikelihood(TabulatedBAOLikelihood):
    """
    eBOSS DR16 Lyman-alpha forest BAO, ``(D_M/r_d, D_H/r_d)`` at
    ``z_eff = 2.334``, as a tabulated two-dimensional likelihood.

    The default version multiplies the forest auto-correlation by
    its cross-correlation with quasars; ``"dr16_auto"`` and
    ``"dr16_cross"`` are the halves.

    Warning
    -------
    Overlaps ``"desi"``, whose own Lyman-alpha measurement at
    ``z ~ 2.33`` is drawn from much of the same sky. Do not use more
    than one version at a time either -- the halves are what the
    default is built from.
    """

    FAMILY = "eboss_lya"

    LABEL = r"eBOSS-DR16-Lya"


class EBOSSELGFullShapeLikelihood(TabulatedBAOLikelihood):
    """
    eBOSS DR16 emission-line galaxies, full shape: the joint
    likelihood of ``(D_M/r_d, D_H/r_d, f sigma_8)`` at
    ``z_eff = 0.845``, on a 100x100x100 grid.

    The same galaxies as :class:`EBOSSELGLikelihood`, analysed
    differently. That one compresses the clustering to a single
    isotropic BAO scale; this one keeps the anisotropy and the
    growth rate as well, so it constrains the amplitude of structure
    directly rather than only the geometry.

    Being three-dimensional is the point. ``f sigma_8`` is degenerate
    with the Alcock-Paczynski distortion, and the grid holds that
    degeneracy exactly, which two separate error bars could not. It
    is also why the prediction here uses ``fsigma8(z)`` unrescaled:
    the geometry the measurement was made against is a *coordinate*
    of the grid, not a fiducial to correct back to, so applying the
    AP correction that :class:`~likelihoods.fsigma8.FSigma8Likelihood`
    needs would count it twice.

    Warning
    -------
    Mutually exclusive with ``"eboss_elg"`` -- the same galaxies,
    twice. Overlaps ``"desi"``, whose ELG sample succeeds this one.
    It also overlaps ``"fsigma8"``, whose compilation includes eBOSS
    growth-rate measurements.
    """

    FAMILY = "eboss_elg_fs"

    LABEL = "eBOSS-DR16-ELG-FS"

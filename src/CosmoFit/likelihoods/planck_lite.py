"""
Planck 2018 ``plik_lite`` high-l TT/TE/EE likelihood.

The CMB, computed rather than compressed
----------------------------------------
:class:`~likelihoods.planck.PlanckLikelihood` reduces Planck to
three numbers (R, l_A, omega_b h^2) and compares them against a
background ``E(z)``. That is fast, dependency-free, works for every
model in the library, and is what most dark-energy papers use.

It also throws almost all of the information away, and -- as that
module's own docstring documents at length -- it inherits every
convention of whoever produced the compression. Get one of those
conventions wrong and the likelihood returns chi2 ~ 100 at Planck's
own best fit, with no symptom other than shifted parameters.

This likelihood is the other end of that trade. It uses the actual
measured bandpowers: 613 binned TT, TE and EE points spanning
l = 30-2508, with their full 613x613 covariance, compared against
a C_l spectrum computed from scratch by a Boltzmann code (see
:mod:`cosmology.boltzmann`). There is no compression, no derived
summary statistic, and no borrowed convention -- the theory
prediction is the same object the data is.

What it costs:

- **A Boltzmann code.** CAMB, as an optional dependency
  (``pip install "cosmofit[cmb]"``).
- **Speed.** One CAMB call is ~0.7 s against ~1 ms for the whole
  rest of a joint likelihood, so a chain that includes this is
  roughly three orders of magnitude slower per step. That is not
  an implementation flaw; it is why full CMB chains are run on
  clusters and why compressed priors exist. Budget hours, use
  ``n_processes``, and save the chain.
- **Model coverage.** Only models CAMB can represent: LCDM, and
  anything exposing a ``w(z)``. The modified-gravity models are
  refused outright rather than silently handed GR perturbations.
  See :class:`~cosmology.boltzmann.CAMBBackend`.
- **More parameters.** A CMB spectrum depends on the primordial
  amplitude and tilt (``ln1e10As``, ``n_s``) and the reionization
  optical depth (``tau_reio``), none of which a background fit ever
  needed. They must be free, or fixed deliberately.

Why ``plik_lite`` and not the full likelihood
---------------------------------------------
Planck's full high-l likelihood carries ~20 nuisance parameters
describing galactic dust, point sources, CIB and SZ contamination.
``plik_lite`` is the variant in which the Planck team have already
marginalized over all of them, leaving one calibration parameter
(``A_planck``). That is what makes it usable outside a full Planck
pipeline, and it is what public reimplementations
(``planck-lite-py``, Cobaya's ``planck_2018_highl_plik.TTTEEE_lite``)
target. The cost is a small loss of information relative to the
full likelihood, and the inability to re-examine the foreground
model.

l < 30 is not included
----------------------
``plik_lite`` starts at l = 30. The low-l temperature and
polarization likelihoods (Commander, SimAll) are separate products
with different statistics -- at low l the C_l distribution is not
Gaussian and a bandpower covariance does not describe it. The
practical consequence is that ``tau`` enters only through the
``exp(-2 tau)`` damping of the high-l spectra and is therefore
degenerate with ``ln1e10As``. Add the ``"tau"`` dataset (Planck's
lowE constraint, ``tau = 0.0544 +- 0.0073``) to break it. Without
it, ``ln1e10As`` and anything derived from it are unconstrained,
and the fit will say so by returning a posterior as wide as the
prior.

Validation
----------
Against the reference spectrum shipped with ``planck-lite-py``
(a CLASS Planck-2015 best fit), this implementation reproduces its
published log-likelihood values exactly -- see
``tests/test_planck_lite.py``.

References
----------
Planck Collaboration (2020), A&A 641, A5, arXiv:1907.12875
(the likelihood).
Planck Collaboration (2020), A&A 641, A6, arXiv:1807.06209
(cosmological parameters).
"""

from __future__ import annotations

import numpy as np

from CosmoFit.cosmology.boltzmann import CAMBBackend
from CosmoFit.data.loader import load_plik_lite
from CosmoFit.likelihoods.covariance import make_covariance

from .base import BaseLikelihood


#: Which spectra each ``spectra=`` selection uses, in the order
#: they appear in the released data vector.
SPECTRA_SETS = {

    "TT": ("TT",),

    "TE": ("TE",),

    "EE": ("EE",),

    "TTTEEE": ("TT", "TE", "EE"),

}


#: LaTeX label for each spectrum, for figure titles.
SPECTRUM_LABELS = {
    "TT": r"$C_\ell^{TT}$",
    "TE": r"$C_\ell^{TE}$",
    "EE": r"$C_\ell^{EE}$",
}


class PlanckLiteLikelihood(BaseLikelihood):
    """
    Planck 2018 ``plik_lite`` binned TT/TE/EE bandpowers against a
    Boltzmann-computed C_l spectrum.

    Parameters
    ----------
    cosmology
        Cosmology model instance. Must be one CAMB can represent
        (LCDM, or any model with a ``w(z)``); anything else raises
        :class:`~cosmology.boltzmann.BoltzmannError` at
        construction rather than mid-chain.

    version : str, optional
        Dataset version.

    spectra : str, optional
        Which spectra to use: ``"TTTEEE"`` (default, all 613
        bandpowers), or ``"TT"`` / ``"TE"`` / ``"EE"`` for one
        alone. Selecting a subset takes the corresponding block of
        the covariance -- correctly dropping, not zeroing, the
        cross-spectrum correlations.

    lens_potential_accuracy : int, optional
        Passed to :class:`~cosmology.boltzmann.CAMBBackend`.

    Warning
    -------
    Do not combine ``"planck_lite"`` with ``"planck"``. They are
    the same Planck data twice -- the distance priors are a
    compression *of these bandpowers* -- and using both counts the
    entire CMB dataset two ways, once in full and once in summary.
    Pick the compression (fast, every model) or the spectra (slow,
    complete), not both.
    """

    def __init__(
        self,
        cosmology,
        version: str = "planck2018",
        spectra: str = "TTTEEE",
        use_low_ell: bool = False,
        lens_potential_accuracy: int = 1,
    ):

        if spectra not in SPECTRA_SETS:

            raise ValueError(

                f"Unknown spectra selection '{spectra}'. "

                f"Available: {list(SPECTRA_SETS)}",

            )

        self.spectra = spectra

        self.used = SPECTRA_SETS[spectra]

        self.use_low_ell = use_low_ell

        dataset = load_plik_lite(version, use_low_ell=use_low_ell)

        # Constructed before `super().__init__` so that an
        # unrepresentable model fails here, at setup, rather than
        # thousands of MCMC steps in.
        # Shared with any other CAMB-based likelihood on the same
        # cosmology -- see `CAMBBackend.shared`.
        self.backend = CAMBBackend.shared(

            cosmology,

            lmax=dataset.lmax,

            lens_potential_accuracy=lens_potential_accuracy,

        )

        self.full_data = dataset

        super().__init__(

            name=(
                f"Planck-lite {spectra}"
                + (" + lowTT" if use_low_ell else "")
            ),

            dataset=self._restrict(dataset),

            cosmology=cosmology,

        )

    # ---------------------------------------------------------

    def _restrict(self, dataset):
        """
        Cut the dataset down to the selected spectra.

        Returns the dataset unchanged for the full ``"TTTEEE"``
        selection, so the common case pays nothing.
        """

        if self.spectra == "TTTEEE":

            self._select = np.arange(dataset.size)

            return dataset

        from dataclasses import replace

        keep = np.concatenate(

            [

                np.arange(

                    dataset.slices[name].start,

                    dataset.slices[name].stop,

                )

                for name in self.used

            ],

        )

        self._select = keep

        cov = dataset.covariance.matrix[np.ix_(keep, keep)]

        counts = tuple(

            (

                dataset.slices[name].stop

                - dataset.slices[name].start

            )

            if name in self.used

            else 0

            for name in SPECTRA_SETS["TTTEEE"]

        )

        return replace(

            dataset,

            ell=dataset.ell[keep],

            value=dataset.value[keep],

            sigma=dataset.sigma[keep],

            covariance=make_covariance(cov=cov),

            n_bin=counts,

        )

    # ---------------------------------------------------------

    @property
    def observable(
        self,
    ):

        return f"C_l ({self.spectra})"

    # ---------------------------------------------------------

    def _bin(
        self,
        cl: np.ndarray,
        spectrum: str,
        cl_lmin: int,
    ) -> np.ndarray:
        """
        Apply Planck's bandpower window functions to a theory C_l
        array.

        Parameters
        ----------
        cl
            C_l from ``lmin`` upward, in muK^2, one entry per
            multipole.

        Notes
        -----
        The windows are stored as ``blmin``/``blmax`` (first and
        last multipole *index*, relative to ``lmin``) plus a flat
        ``weights`` array indexed by the same offset -- and the
        ranges are inclusive of ``blmax``, a Fortran convention
        that costs a ``+1`` here and is the classic off-by-one in
        reimplementations of this likelihood. TT, TE and EE share
        one set of windows; only the number of bandpowers differs.
        """

        data = self.full_data

        if spectrum == "TT":

            blmin, blmax, weights, lmin_windows = data.tt_windows

            n_max = data.n_bin[0]

        else:

            blmin, blmax, weights, lmin_windows = (
                data.blmin, data.blmax, data.weights, data.lmin
            )

            n_max = max(data.n_bin[1:])

        binned = np.empty(n_max, dtype=float)

        for i in range(n_max):

            # Window i covers multipoles
            # [blmin[i] + lmin_windows, blmax[i] + lmin_windows],
            # and `cl` is indexed from `cl_lmin` -- so the two
            # offsets do not cancel in general and both have to be
            # written out. The weights, by contrast, are indexed by
            # the window offset alone.
            lo = blmin[i] + lmin_windows - cl_lmin
            hi = blmax[i] + lmin_windows - cl_lmin + 1

            binned[i] = np.dot(

                cl[lo:hi],

                weights[blmin[i]:blmax[i] + 1],

            )

        return binned

    # ---------------------------------------------------------

    def model(
        self,
    ) -> np.ndarray:
        """
        Predicted bandpowers for the current cosmology.

        Runs the Boltzmann code, bins the resulting C_l with
        Planck's own window functions, and divides by
        ``A_planck^2``.
        """

        data = self.full_data

        # With the low-l bins the TT windows reach down to l = 2, so
        # the theory spectra have to start there rather than at
        # plik_lite's l = 30.
        cl_lmin = min(data.lmin, data.lmin_tt)

        cls = self.backend.cls(lmin=cl_lmin)

        n_tt, n_te, n_ee = data.n_bin

        counts = {"TT": n_tt, "TE": n_te, "EE": n_ee}

        prediction = np.concatenate(

            [

                self._bin(cls[name], name, cl_lmin)[: counts[name]]

                for name in SPECTRA_SETS["TTTEEE"]

            ],

        )

        prediction = prediction / self.cosmology.A_planck ** 2

        return prediction[self._select]

    # ---------------------------------------------------------

    def residuals(
        self,
    ) -> np.ndarray:
        """
        Data minus model bandpower residuals.
        """

        return (

            self.data.value

            - self.model()

        )

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Chi-square statistic.
        """

        return self.covariance.chi2(

            self.residuals(),

        )

r"""
DESI DR1 full-shape clustering, compressed with ShapeFit.

DESI 2024 V (arXiv:2411.12021) fits the power spectrum of each of its
six DR1 tracers with four compressed parameters, and its appendix A
gives them, with their Gaussian covariance, as

    D_V / r_d,   D_H / D_M,   f sigma_s8,   m + n.

The first two are the BAO scale and the Alcock-Paczynski ratio; the
third is the growth of structure. The fourth, the shape of the power
spectrum, is left out here -- marginalized, which for a Gaussian is
dropping its row and column -- since predicting it takes the
no-wiggle slope of the linear spectrum at a pivot, a model of its own.

``f sigma_s8`` is the one subtle quantity. ``sigma_s8`` is the r.m.s.
of the cold matter in spheres of ``s8 = 8 Mpc/h`` *of the fiducial
cosmology*; in a cosmology whose sound horizon differs from the
fiducial one, the template's scales are stretched by ``r_d /
r_d^fid``, so the radius is

    R = s8 r_d / r_d^fid = 8 r_d / 99.0792  [Mpc],

``h`` of the fiducial cosmology cancelling (``r_d^fid = 99.0792
Mpc/h``, DESI 2024 V, table 11). The prediction is then the growth
rate times ``sigma(R)`` of the cold matter, read off CAMB. At DESI's
fiducial cosmology this gives back the paper's table 11 -- its
``sigma_s8`` and ``f sigma_s8`` -- to the four digits it prints (the
tests check it).

The tracers are independent of each other, as the paper treats them.

Versions: ``shapefit`` (the full-shape fits alone, the default) and
``shapefit_bao`` (combined with DESI's post-reconstruction BAO, which
are the same galaxies as the DR1 BAO). Neither may be combined with
DESI BAO, of either release: DR2 contains DR1's galaxies.

Needs the ``camb`` theory and ``growth`` with ``amplitude: boltzmann``.
"""

from __future__ import annotations

import warnings

import numpy as np

from scipy.interpolate import CubicSpline

from cosmofit.core.component import ComponentError, Likelihood
from cosmofit.data.loaders._common import DATA_DIR


__all__ = ["DESIFullShape", "load_desi_shapefit"]


#: ``r_d`` of DESI 2024 V's fiducial cosmology, in its Mpc/h.
RD_FIDUCIAL = 99.0792

#: The smoothing scale of ``sigma_s8``, in the fiducial Mpc/h.
S8_RADIUS = 8.0

#: Radii [Mpc] at which CAMB's ``sigma(R)`` is tabulated, interpolated
#: in between: ``r_d`` from 87 to 248 Mpc, far wider than any prior.
_RADII = np.geomspace(7.0, 20.0, 41)

VERSIONS = {
    "shapefit": "desi_dr1_shapefit.txt",
    "shapefit_bao": "desi_dr1_shapefit_bao.txt",
}


def load_desi_shapefit(version: str = "shapefit") -> dict:
    """
    The data of one version: ``tracers``, ``z``, the data vectors
    ``(D_V/r_d, D_H/D_M, f sigma_s8)`` and their 3x3 covariances --
    ``m + n`` dropped.
    """

    if version not in VERSIONS:
        raise ValueError(
            f"Unknown DESI full-shape version {version!r}; known: {sorted(VERSIONS)}."
        )

    path = DATA_DIR / "bao" / "desi_dr1_shapefit" / VERSIONS[version]

    tracers, rows = [], []

    for line in path.read_text(encoding="utf-8").splitlines():

        if not line.strip() or line.startswith("#"):
            continue

        name, *numbers = line.split()

        tracers.append(name)
        rows.append([float(x) for x in numbers])

    rows = np.array(rows)

    z = rows[:, 0]
    vectors = rows[:, 1:5]
    upper = rows[:, 5:15] * 1.0e-4

    covariances = np.empty((len(z), 4, 4))
    i, j = np.triu_indices(4)

    for n in range(len(z)):
        covariances[n][i, j] = upper[n]
        covariances[n][j, i] = upper[n]

    return {
        "tracers": tracers,
        "z": z,
        "data": vectors[:, :3],
        "covariance": covariances[:, :3, :3],
    }


class DESIFullShape(Likelihood):
    """
    Options
    -------
    version : str
        ``"shapefit"`` (the default) or ``"shapefit_bao"``.
    rd : str
        ``"computed"`` (the default; from the ``early_universe``
        theory) or ``"free"`` (a parameter ``rd`` [Mpc]).
    """

    def initialize(self) -> None:

        unknown = set(self.info) - {"version", "rd"}

        if unknown:
            raise ComponentError(f"{self.name}: unknown option(s) {sorted(unknown)}.")

        self.version = self.info.get("version", "shapefit")
        self.rd = self.info.get("rd", "computed")

        if self.rd not in ("computed", "free"):
            raise ComponentError(
                f"{self.name}: rd must be 'computed' or 'free', not {self.rd!r}."
            )

        try:
            data = load_desi_shapefit(self.version)
        except ValueError as error:
            raise ComponentError(f"{self.name}: {error}") from None

        self.tracers = data["tracers"]
        self.z = data["z"]
        self.data = data["data"]
        self.inverse = np.linalg.inv(data["covariance"])

    def accepts(self, name: str) -> bool:
        return name == "rd" and self.rd == "free"

    def get_default_params(self) -> dict:
        return {}

    def get_required_params(self) -> list[str]:
        return ["rd"] if self.rd == "free" else []

    def get_requirements(self) -> dict:

        requirements = {
            "DM": None, "DH": None, "growth_rate": None,
            "sigma_R": {
                "z": list(self.z), "R": list(_RADII),
                "vars_pairs": [["delta_nonu", "delta_nonu"]],
            },
        }

        if self.rd == "computed":
            requirements["rdrag"] = None

        return requirements

    def check_model(self, model) -> None:

        from cosmofit.likelihoods.native import DESI

        for other in model.likelihoods.values():
            if isinstance(other, DESI):
                warnings.warn(
                    f"Likelihoods '{self.name}' and '{other.name}' should not be "
                    f"combined: the DESI full-shape fits are of DESI DR1's "
                    f"galaxies, which DESI's BAO (DR1 and DR2) also measure. "
                    f"Treating them as independent double-counts that data. "
                    f"Use '{self.name}' with version 'shapefit_bao' instead, "
                    f"or DESI BAO alone.",
                    UserWarning,
                    stacklevel=2,
                )

    # ---------------------------------------------------------

    def predictions(self, rd: float | None = None) -> np.ndarray:
        """``(D_V/r_d, D_H/D_M, f sigma_s8)`` for each tracer."""

        provider = self.provider

        if rd is None:
            rd = provider.get_rdrag()

        z = self.z

        DM = np.asarray(provider.get_DM(z), dtype=float)
        DH = np.asarray(provider.get_DH(z), dtype=float)

        DV = np.cbrt(z * DM ** 2 * DH)

        radius = S8_RADIUS * rd / RD_FIDUCIAL

        if not _RADII[0] <= radius <= _RADII[-1]:
            raise ComponentError(
                f"{self.name}: r_d = {rd:g} Mpc puts sigma_s8's radius at "
                f"{radius:g} Mpc, outside the {_RADII[0]:g}..{_RADII[-1]:g} "
                f"tabulated."
            )

        zs, radii, sigma = provider.get_sigma_R(("delta_nonu", "delta_nonu"))

        sigma_s8 = np.empty(len(z))

        for n, redshift in enumerate(z):

            row = int(np.argmin(np.abs(zs - redshift)))

            spline = CubicSpline(np.log(radii), np.log(sigma[row]))
            sigma_s8[n] = np.exp(spline(np.log(radius)))

        f = np.asarray(provider.get_growth_rate(z), dtype=float)

        return np.column_stack([DV / rd, DH / DM, f * sigma_s8])

    def logp(self, **params) -> float:

        residual = self.data - self.predictions(params.get("rd"))

        chi2 = np.einsum("ni,nij,nj->", residual, self.inverse, residual)

        return -0.5 * float(chi2)

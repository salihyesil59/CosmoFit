"""
Cosmological models.

This subpackage contains concrete implementations of the
:class:`~cosmology.core.base.Cosmology` base class, in four
families:

**Dark energy on top of GR** -- the expansion history is changed by
adding or reshaping a fluid, but gravity is Einstein's:
:class:`~cosmology.models.lcdm.LCDM`,
:class:`~cosmology.models.wcdm.WCDM`,
:class:`~cosmology.models.cpl.CPL`,
:class:`~cosmology.models.jbp.JBP`,
:class:`~cosmology.models.ba.BA`,
:class:`~cosmology.models.logarithmic.LogarithmicDE`,
:class:`~cosmology.models.pede.PEDE`,
:class:`~cosmology.models.gede.GEDE`,
:class:`~cosmology.models.lscdm.LsCDM`.

**Unified or interacting dark sector** -- one fluid does both jobs,
or the two exchange energy: :class:`~cosmology.models.gcg.GCG`,
:class:`~cosmology.models.ide.IDE`,
:class:`~cosmology.models.rvm.RunningVacuum`.

**Modified Friedmann equation** -- acceleration without dark energy,
from a changed relation between H and rho:
:class:`~cosmology.models.cardassian.Cardassian`,
:class:`~cosmology.models.dgp.DGP`.

**Modified gravity** -- the field equations themselves differ:
:class:`~cosmology.models.fq.FQExponential`,
:class:`~cosmology.models.frt.FRTLinear`,
:class:`~cosmology.models.fr.FRHuSawicki`.

Which family a model belongs to decides what can be done with it.
Everything here has an ``E(z)``, so every dataset that is a
background probe works for all of them. Only the models with a
``w(z)`` can be handed to a Boltzmann code for a from-scratch CMB
spectrum (see :class:`~cosmology.boltzmann.CAMBBackend`). Only the
models that override ``mu(a, k)`` -- the three modified-gravity ones
and DGP -- predict a growth history that differs from GR's at a
fixed background.
"""

from .lcdm import LCDM
from .wcdm import WCDM
from .cpl import CPL
from .jbp import JBP
from .ba import BA
from .logarithmic import LogarithmicDE
from .pede import PEDE
from .gede import GEDE
from .lscdm import LsCDM
from .gcg import GCG
from .ide import IDE
from .rvm import RunningVacuum
from .cardassian import Cardassian
from .dgp import DGP
from .fq import FQExponential
from .frt import FRTLinear
from .fr import FRHuSawicki

__all__ = [
    "LCDM",
    "WCDM",
    "CPL",
    "JBP",
    "BA",
    "LogarithmicDE",
    "PEDE",
    "GEDE",
    "LsCDM",
    "GCG",
    "IDE",
    "RunningVacuum",
    "Cardassian",
    "DGP",
    "FQExponential",
    "FRTLinear",
    "FRHuSawicki",
]

from .parameters import CosmologyParameters
from .integrals import DistanceIntegrator
from .distances import DistanceCalculator
from .background import BackgroundCalculator
from .sound_horizon import SoundHorizon


class Cosmology:
    """
    Base cosmology class.
    """

    def __init__(self, params: CosmologyParameters):

        self.params = params

        self.integrator = DistanceIntegrator(self)

        self.distance = DistanceCalculator(self)

        self.background = BackgroundCalculator(self)

        self.sound_horizon = SoundHorizon(self)

    # ---------------------------------------------------------

    @property
    def H0(self):
        return self.params.H0

    @property
    def Omega_m(self):
        return self.params.Omega_m

    @property
    def Omega_k(self):
        return self.params.Omega_k

    @property
    def w0(self):
        return self.params.w0

    @property
    def wa(self):
        return self.params.wa

    @property
    def rd(self):
        return self.params.rd

    # ---------------------------------------------------------

    @property
    def Omega_de0(self):
        return 1.0 - self.Omega_m - self.Omega_k

    # ---------------------------------------------------------

    def E(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def dEdz(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def Omega_de(self, z):
        raise NotImplementedError

    # ---------------------------------------------------------

    def H(self, z):
        return self.H0 * self.E(z)
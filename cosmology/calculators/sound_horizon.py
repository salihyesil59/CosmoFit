"""
Sound horizon calculations.

Currently rd is treated as an external parameter.
Future versions may implement Eisenstein-Hu or
Boltzmann-code based calculations.
"""


class SoundHorizon:

    def __init__(self, cosmology):

        self.cosmo = cosmology

    @property
    def rd(self):

        return self.cosmo.rd
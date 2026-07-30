from dataclasses import dataclass

@dataclass(slots=True)
class CosmologyParameters:
    """
    Container for cosmological parameters.
    """

    H0: float
    Omega_m: float

    w0: float = -1.0
    wa: float = 0.0

    rd: float = 147.0

    Omega_k: float = 0.0
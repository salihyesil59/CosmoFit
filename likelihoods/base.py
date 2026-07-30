"""
Base likelihood class.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from .covariance import make_covariance


class Likelihood(ABC):
    """
    Abstract base class for cosmological likelihoods.
    """

    def __init__(
        self,
        name: str,
        dataset,
        calculator,
    ):
        """
        Parameters
        ----------
        name
            Likelihood name.

        dataset
            Observational dataset.

        calculator
            BackgroundCalculator instance.
        """

        self.name = name

        self.data = dataset

        self.calculator = calculator

        if getattr(
            dataset,
            "covariance",
            None,
        ) is None:

            self.covariance = make_covariance(
                sigma=dataset.sigma,
            )

        else:

            self.covariance = make_covariance(
                cov=dataset.covariance,
            )

    # ========================================================
    # Properties
    # ========================================================

    @property
    def n_data(
        self,
    ) -> int:
        """
        Number of data points.
        """

        return self.data.size

    @property
    def name_and_size(
        self,
    ) -> str:
        """
        Name together with the number of data points.
        """

        return f"{self.name} ({self.n_data})"

    # ========================================================
    # Abstract interface
    # ========================================================

    @abstractmethod
    def model(
        self,
    ):
        """
        Return theoretical predictions.
        """

        pass

    @abstractmethod
    def chi2(
        self,
    ) -> float:
        """
        Return chi-square.
        """

        pass

    # ========================================================
    # Common methods
    # ========================================================

    def predictions(
        self,
    ):
        """
        Alias for model().
        """

        return self.model()

    def log_likelihood(
        self,
    ) -> float:
        """
        Return log-likelihood.
        """

        return -0.5 * self.chi2()

    def summary(
        self,
    ) -> dict:
        """
        Return a summary of the likelihood evaluation.
        """

        chi2 = self.chi2()

        return {

            "name": self.name,

            "n_data": self.n_data,

            "chi2": chi2,

            "loglike": self.log_likelihood(),

        }

    # ========================================================
    # Representation
    # ========================================================

    def __str__(
        self,
    ) -> str:
        """
        Human-readable representation.
        """

        return (
            f"{self.__class__.__name__}"
            f"(name='{self.name}', "
            f"n_data={self.n_data})"
        )

    __repr__ = __str__
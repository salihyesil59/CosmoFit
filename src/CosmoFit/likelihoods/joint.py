"""
Joint likelihood.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .base import BaseLikelihood


class JointLikelihood:
    """
    Joint likelihood composed of multiple independent likelihoods.
    """

    def __init__(
        self,
        *likelihoods: BaseLikelihood,
    ) -> None:

        self.likelihoods = list(

            likelihoods,

        )

    # ---------------------------------------------------------

    def __len__(
        self,
    ) -> int:

        return len(

            self.likelihoods,

        )

    # ---------------------------------------------------------

    def __iter__(
        self,
    ) -> Iterator[BaseLikelihood]:

        return iter(

            self.likelihoods,

        )

    # ---------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        names = ", ".join(

            likelihood.name

            for likelihood in self

        )

        return (

            f"{self.__class__.__name__}"

            f"({names})"

        )

    # ---------------------------------------------------------

    def add(
        self,
        likelihood: BaseLikelihood,
    ) -> None:

        self.likelihoods.append(

            likelihood,

        )

    # ---------------------------------------------------------

    @property
    def n_data(
        self,
    ) -> int:

        return sum(

            likelihood.n_data

            for likelihood in self

        )

    # ---------------------------------------------------------

    @property
    def names(
        self,
    ) -> list[str]:

        return [

            likelihood.name

            for likelihood in self

        ]

    # ---------------------------------------------------------

    def chi2(
        self,
    ) -> float:
        """
        Total chi-square.
        """

        return sum(

            likelihood.chi2()

            for likelihood in self

        )

    # ---------------------------------------------------------

    def loglike(
        self,
    ) -> float:
        """
        Total log-likelihood.
        """

        return -0.5 * self.chi2()

    # ---------------------------------------------------------

    def summary(
        self,
    ) -> dict:
        """
        Summary of the joint likelihood.
        """

        summaries = [

            likelihood.summary()

            for likelihood in self

        ]

        return {

            "n_likelihoods": len(

                self,

            ),

            "n_data": self.n_data,

            "chi2": self.chi2(),

            "loglike": self.loglike(),

            "likelihoods": summaries,

        }

    # ---------------------------------------------------------

    def aic(
        self,
        n_params: int,
    ) -> float:
        """
        Akaike Information Criterion.
        """

        return (

            self.chi2()

            +

            2.0 * n_params

        )

    # ---------------------------------------------------------

    def bic(
        self,
        n_params: int,
    ) -> float:
        """
        Bayesian Information Criterion.
        """

        return (

            self.chi2()

            +

            n_params

            * np.log(

                self.n_data,

            )

        )

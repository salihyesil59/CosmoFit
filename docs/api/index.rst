API reference
=============

Everything importable from ``CosmoFit`` itself is on this page; the
subpackages below carry the rest.

.. currentmodule:: CosmoFit

Models
------

Twenty expansion histories written out by hand, and three routes to
one that is not here. Every one is a :class:`Cosmology` subclass, so
every dataset, likelihood, sampler and plot works on all of them
without knowing which.

.. autosummary::
   :toctree: generated
   :nosignatures:

   Cosmology
   CosmologyParameters
   ModelConfigurationError
   LCDM
   WCDM
   CPL
   JBP
   BA
   LogarithmicDE
   PEDE
   GEDE
   LsCDM
   GCG
   IDE
   RunningVacuum
   Cardassian
   HDE
   ADE
   RDE
   DGP
   FQExponential
   FRTLinear
   FRHuSawicki
   define_model
   model_from_expression

Likelihoods
-----------

One class per dataset, plus the joint likelihood that combines them.
Three of these are tabulated rather than Gaussian -- they carry a
released likelihood *surface* instead of a mean and a covariance.

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseLikelihood
   JointLikelihood
   CCLikelihood
   DESILikelihood
   SDSSBAOLikelihood
   SDSSFullShapeLikelihood
   EBOSSELGLikelihood
   EBOSSELGFullShapeLikelihood
   EBOSSLyaLikelihood
   BAOLowZLikelihood
   PantheonLikelihood
   DESSN5YRLikelihood
   Union3Likelihood
   PlanckLikelihood
   PlanckLiteLikelihood
   PlanckLensingLikelihood
   PlanckLowEELikelihood
   ACTDR6LensingLikelihood
   FSigma8Likelihood
   S8Likelihood
   H0Likelihood
   OmegaBLikelihood
   TauLikelihood

Fitting
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Fitter
   BaseSampler
   EnsembleSampler
   FitResult
   BestFitResult
   MCMCResult

Plotting
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   FitPlotter

Saved chains
------------

A chain is written as it is sampled, resumed rather than re-sampled
next time, and can be reopened months later without loading a single
dataset.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ChainFile
   StoredSampler
   open_chain
   chain_info

Datasets
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   available_datasets
   available_versions
   dataset_reference

Subpackages
-----------

.. toctree::
   :maxdepth: 1

   cosmology
   likelihoods
   stats
   theory

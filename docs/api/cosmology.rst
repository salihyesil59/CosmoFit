``CosmoFit.cosmology``
======================

.. automodule:: CosmoFit.cosmology

Calculators
-----------

The pieces every model shares: the background evolution, the distance
integrals, recombination, the sound horizon and linear growth. A model
supplies ``E(z)``; these supply everything derived from it.

.. autosummary::
   :toctree: generated
   :nosignatures:

   BackgroundCalculator
   DistanceCalculator
   SoundHorizon
   RecombinationCalculator
   GrowthCalculator
   DistanceIntegrator

Custom models
-------------

.. automodule:: CosmoFit.cosmology.custom
   :members:

The Boltzmann backend
---------------------

.. automodule:: CosmoFit.cosmology.boltzmann
   :members:

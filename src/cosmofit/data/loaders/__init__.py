"""
The dataset loaders, one module per family of probes.

``_common``
    The data directory, version lookup, and the readers every family
    shares.
``cc``, ``bao``, ``sn``, ``growth``, ``cmb``, ``priors``
    Each family's file tables (``<NAME>_FILES``: every version, its
    files and its reference) and its ``load_<dataset>`` functions.
``_index``
    Every family's tables together: :func:`available_versions`,
    :func:`dataset_reference`, :func:`available_datasets`.

:mod:`data.loader` re-exports all of it under the one name it has
always had.
"""

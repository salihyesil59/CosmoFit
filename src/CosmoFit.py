"""
The package's name before 2.0, kept for one release.

The library is imported as ``cosmofit`` -- the name it has always had
on PyPI. ``import CosmoFit`` still works, and so does every submodule
under it (``from CosmoFit.stats import ...``), but each name resolves
to the ``cosmofit`` module itself rather than to a copy: a class
imported one way is the class imported the other, and a chain or a
pickle written under the old name reads back under the new.

This is a module rather than a directory on purpose. On a
case-insensitive filesystem -- Windows, macOS -- a directory
``CosmoFit`` and a directory ``cosmofit`` are one directory.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings


_OLD = "CosmoFit"
_NEW = "cosmofit"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Finds ``CosmoFit.<x>`` as the module ``cosmofit.<x>``."""

    def find_spec(self, fullname, path=None, target=None):

        if fullname.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(fullname, self)

        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD):])

    def exec_module(self, module):
        """Already executed, under its own name."""


warnings.warn(
    "The package is now imported as 'cosmofit'; 'CosmoFit' is an alias for "
    "it that will be removed in a later release.",
    DeprecationWarning,
    stacklevel=2,
)

if not any(isinstance(finder, _Alias) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Alias())

sys.modules[__name__] = importlib.import_module(_NEW)

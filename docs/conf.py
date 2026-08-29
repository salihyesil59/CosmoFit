"""
Sphinx configuration.

The site is an *API reference*, not a second manual. The README and
the seventeen notebooks under ``examples/`` are the narrative
documentation and are better at it; what neither of them gives is a
place to look up one class or one function, with its parameters, its
defaults, and the paper it implements.

So: `autosummary` over the public API, `napoleon` because every
docstring here is numpydoc, and `intersphinx` so that a parameter
annotated ``np.ndarray`` links to numpy rather than sitting there as
text.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ============================================================
# Project
# ============================================================

project = "CosmoFit"
author = "Salih Yeşil"
copyright = "2026, Salih Yeşil"

try:
    release = version("cosmofit")
except PackageNotFoundError:  # pragma: no cover - a bare checkout
    release = "0.0.0+unknown"

version = release


# ============================================================
# Extensions
# ============================================================

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# The README and CHANGELOG are Markdown and are included verbatim, so
# the site cannot drift from them.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = ["colon_fence", "deflist"]

# Both files are long and heavily sectioned; without this only the
# top-level headings are anchors, and every in-page link breaks.
myst_heading_anchors = 3

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The README and CHANGELOG are included verbatim so the site cannot
# drift from them, and they are written for GitHub -- their relative
# links (`CHANGELOG.md`, `examples/`) resolve there and not here.
# Rewriting them for Sphinx would break them where most people read
# them.
suppress_warnings = ["myst.xref_missing"]


# ============================================================
# autodoc
# ============================================================

autosummary_generate = True

autodoc_member_order = "bysource"

autodoc_typehints = "description"

# Every module here uses `from __future__ import annotations`, so
# without this the signatures render as the string form.
autodoc_type_aliases = {}

# Deliberately no `"members": True` here. It would apply to every
# bare `automodule` directive -- including the ones that exist only
# to pull in a module's *narrative* docstring -- and each of those
# would then re-document classes the autosummary pages already own,
# giving two pages per class and an ambiguous cross-reference for
# every name.
autodoc_default_options = {
    "show-inheritance": True,
}

# `CosmoFit.theory` needs sympy, and the CMB path needs CAMB. Neither
# is a hard dependency, so the docs build must not fail without them.
autodoc_mock_imports = ["camb"]

napoleon_google_docstring = False

napoleon_numpy_docstring = True

napoleon_use_rtype = False


# ============================================================
# intersphinx
# ============================================================

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable/", None),
}


# ============================================================
# HTML
# ============================================================

html_theme = "furo"

html_title = f"CosmoFit {release}"

html_static_path = []

html_theme_options = {
    "source_repository": "https://github.com/salihyesil59/CosmoFit/",
    "source_branch": "dev",
    "source_directory": "docs/",
}

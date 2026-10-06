"""
The graphical interface: a Streamlit page over the library.

``cosmofit gui`` starts it (:func:`main`); it needs the ``gui`` extra,
``pip install "cosmofit[gui]"``.

:mod:`~gui.app`
    The page, which Streamlit runs top to bottom on every interaction.
:mod:`~gui.reference`
    What the page knows without computing: models, datasets, presets,
    parameters, plot labels.
:mod:`~gui.helpers`
    The sidebar's choices made into a fit.
:mod:`~gui.render`
    A finished fit drawn.

Importing this package does not import Streamlit; the modules beside
it do.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


__all__ = ["APP", "main"]


#: The page's file, which Streamlit runs.
APP = Path(__file__).with_name("app.py")

#: Theme and privacy settings, passed on the command line so they hold
#: wherever the GUI is started from -- a `.streamlit/config.toml`
#: is only read from the working directory. The accent matches the
#: library's own figures (plots/plotter.py's COLOR_MODEL).
_SETTINGS = (
    "--theme.primaryColor=#d1495b",
    "--theme.font=sans serif",
    "--browser.gatherUsageStats=false",
)


def main(args=()) -> int:
    """
    Start the GUI with Streamlit; ``args`` go to ``streamlit run``
    (``--server.port 8600``, say). Returns Streamlit's exit status.
    """

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            'The GUI needs Streamlit: pip install "cosmofit[gui]".', file=sys.stderr,
        )
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(APP), *_SETTINGS, *args]

    return subprocess.call(command)

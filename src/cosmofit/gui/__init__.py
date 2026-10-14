"""
The graphical interface: a Streamlit page over the library.

``cosmofit gui`` starts it (:func:`main`); it needs the ``gui`` extra,
``pip install "cosmofit[gui]"``.

:mod:`~gui.app`
    The page, which Streamlit runs top to bottom on every interaction:
    the sidebar and a row of tabs, of which only the open one is drawn.
:mod:`~gui.steps`
    The tabs, one module each: Start, Data, Model, Run, Results, Guide.
:mod:`~gui.state`
    The configuration, read from session state as one plan -- what the
    Run tab runs whichever tab set it.
:mod:`~gui.reference`
    What the page knows without computing: models, datasets, presets,
    parameters, plot labels, and the guidance written for each step.
:mod:`~gui.helpers`
    The choices made into a fit: model classes, relevance, warnings.
:mod:`~gui.render`
    A finished fit drawn.
:mod:`~gui.style`
    The page's look and the pieces every tab draws the same way.

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
#: library's own figures (plots/plotter.py's COLOR_MODEL). The file
#: watcher is off: it exists to rerun a page whose source is being
#: edited, and walking the library's modules to watch them only
#: delays a start. The toolbar keeps what a user needs and drops the
#: developer menu.
_SETTINGS = (
    "--theme.primaryColor=#d1495b",
    "--theme.font=sans serif",
    "--browser.gatherUsageStats=false",
    "--server.fileWatcherType=none",
    "--client.toolbarMode=minimal",
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

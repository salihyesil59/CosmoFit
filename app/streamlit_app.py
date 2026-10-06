"""
The GUI, run from the repository: ``streamlit run app/streamlit_app.py``.

The page lives in the package, at :mod:`cosmofit.gui.app`, so that an
installed library has it too (``cosmofit gui``). This file runs it
afresh each time Streamlit reruns the script, which an import would
not: a module is executed once, on its first import.
"""

import runpy

from cosmofit.gui import APP


runpy.run_path(str(APP), run_name="__main__")

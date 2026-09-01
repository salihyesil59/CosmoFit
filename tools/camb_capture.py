"""
pytest plugin: record every CAMB call CosmoFit makes, exactly.

Lives outside the repository on purpose -- it is a diagnostic, not part
of the library. Load with:

    PYTHONPATH=<this directory> CAMB_CAPTURE=<logfile> \
        python -m pytest tests/ -p camb_capture

For each call it pickles the `CAMBparams` object that was handed to
CAMB, together with the lmax asked of `get_lens_potential_cls` and
whether the result came back finite. `camb_replay.py` then reissues
exactly those calls, in order, with no CosmoFit in the process.

That is the question this is built to answer: is the failure a property
of what CAMB is asked, or of the process it is asked in?
"""

import os
import pickle

LOGFILE = os.environ.get("CAMB_CAPTURE")


def pytest_configure(config):

    if not LOGFILE:
        return

    import numpy as np

    from CosmoFit.cosmology.boltzmann import CAMBBackend

    original = CAMBBackend._run_once
    counter = {"n": 0}
    current = {"test": None}
    config._camb_capture_current = current

    # Truncate any previous log.
    open(LOGFILE, "wb").close()

    def capturing_run_once(self):

        counter["n"] += 1
        index = counter["n"]

        pars = self._build_params()

        def write(finite, failed):

            record = {
                "index": index,
                "lmax": self.lmax,
                "lens_potential_accuracy": self.lens_potential_accuracy,
                "model": type(self.cosmo).__name__,
                "test": current["test"],
                "pars": pars,
                "finite": finite,
                "raised": failed,
            }

            # A diagnostic must never be able to fail the thing it is
            # measuring. Params built through DarkEnergyPPF's
            # set_w_a_table do not pickle -- that cost two tests
            # before this guard existed -- so record what can be
            # recorded and mark the rest.
            try:
                blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as exc:  # noqa: BLE001 - diagnostic
                record["pars"] = None
                record["unpicklable"] = f"{type(exc).__name__}: {exc}"
                blob = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)

            with open(LOGFILE, "ab") as fh:
                fh.write(blob)

        try:
            spectra = original(self)

        except Exception as exc:  # noqa: BLE001 - diagnostic
            write(None, f"{type(exc).__name__}: {exc}")
            # Re-raise the original, not a copy: swapping the type
            # here made two tests fail for reasons that had nothing
            # to do with what is being measured.
            raise

        write(
            all(
                bool(np.all(np.isfinite(value)))
                for name, value in spectra.items()
                if name != "ell"
            ),
            None,
        )

        return spectra

    CAMBBackend._run_once = capturing_run_once

    config._camb_capture_counter = counter


def pytest_unconfigure(config):

    counter = getattr(config, "_camb_capture_counter", None)

    if counter is not None:
        print(f"\n[camb_capture] recorded {counter['n']} CAMB calls "
              f"to {LOGFILE}")


def pytest_runtest_setup(item):

    current = getattr(item.config, "_camb_capture_current", None)

    if current is not None:
        current["test"] = item.nodeid

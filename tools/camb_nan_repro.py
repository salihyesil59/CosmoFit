"""
CAMB returns a NaN lensing potential after matplotlib lays out text.

On Windows, in a process that has imported `scipy.stats` and has
laid out a single string through matplotlib's FreeType binding,
CAMB's non-linear lensing calculation returns NaN. Everything else
CAMB computes -- background, transfer functions, P(k), sigma8, and
the *unlensed* CMB spectra -- is correct in the same call. Nothing
raises; the NaN is returned as an ordinary result.

    python camb_nan_repro.py            # one run, exit 1 on NaN
    python camb_nan_repro.py 10         # ten runs, report the rate

Each run must be a fresh process: the outcome is decided once per
process and repeating the call inside one process gives the same
answer every time.

What the three ingredients have to be
-------------------------------------
All three are required; any two of them are harmless. Measured over
fresh processes -- 10 for the first row, 8 for the rest -- on the
versions printed by this script:

    scipy.stats + FreeType layout + CAMB      7/10 NaN
    FreeType layout + CAMB, no scipy          0/8
    scipy.stats + CAMB, no text laid out      0/8
    scipy.linalg (or special, integrate,      0/8 each
      interpolate, optimize, sparse)
      + FreeType layout + CAMB

`scipy.stats` is the only submodule that does it, which rules out
scipy's bundled OpenBLAS -- `scipy.linalg` is what loads that, and
it is clean.

**The order matters.** Importing `scipy.stats` *after* the glyph
layout instead of before it is clean, 0/8. So this is not simply
"both are present in the process"; it depends on which is
initialised first, which is what one would expect from a
module-layout or DLL-binding effect rather than from anything either
library computes.

Thread settings are not needed and not the cause: with
`OMP_NUM_THREADS` unset this is 8/8 on a 16-core machine, and with
`OMP_NUM_THREADS=1` it is 7/10. The rates quoted above were measured
with `OMP_NUM_THREADS=1`.

It is also specific to the non-linear branch: setting
`pars.NonLinear = camb.model.NonLinear_none` makes the same call
finite. HMcode's own feedback output is byte-identical between a
failing run and a clean one, and the non-linear P(k) it produces is
finite, so the damage is not in deriving the correction.

Not the floating-point mode: MXCSR and the CRT control word are
unchanged (0x1fa0, round-to-nearest, all exceptions masked) in
every failing run. Not memory pressure either: substituting
allocation churn for the text layout -- small numpy blocks, large
numpy blocks, or Python objects -- is clean 0/8 in all three cases.
"""

import sys

import numpy as np


FONT = "DejaVu Sans"


def lay_out_one_string():
    """
    The trigger. `FT2Font.set_text` is as small as it gets: no
    figure, no renderer, no pyplot. Merely opening the font is not
    enough -- the glyphs have to be laid out.
    """

    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    font = FT2Font(font_manager.findfont(FONT))

    font.set_size(10, 100)

    font.set_text("Redshift z")

    font.get_width_height()


def ask_camb():
    """
    A standard Planck-like LCDM call with CMB lensing switched on.
    Returns a dict of {name: is_finite}.
    """

    import camb

    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=70.0, ombh2=0.02237, omch2=0.1177, tau=0.0544, mnu=0.06,
    )

    pars.InitPower.set_params(As=np.exp(3.044) / 1e10, ns=0.9649)

    # lens_potential_accuracy > 0 is what turns the non-linear
    # correction on; with it at 0, or NonLinear_none, this is clean.
    pars.set_for_lmax(2508, lens_potential_accuracy=1)

    # Only so that sigma8 can be shown alongside, as evidence that
    # the damage is confined to the lensing branch.
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)

    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)

    def ok(x):
        return bool(np.all(np.isfinite(np.asarray(x, dtype=float))))

    return {
        "sigma8": ok([results.get_sigma8_0()]),
        "unlensed_scalar": ok(powers["unlensed_scalar"]),
        "lens_potential": ok(powers["lens_potential"]),
        "lensed_scalar": ok(powers["lensed_scalar"]),
    }


def one_run(verbose=True):

    import scipy.stats  # noqa: F401  -- ingredient one

    lay_out_one_string()  # ingredient two

    state = ask_camb()  # ingredient three

    if verbose:
        for name, good in state.items():
            print(f"  {name:18s} {'finite' if good else 'NaN'}")

    return state["lens_potential"]


def versions():

    import camb
    import matplotlib
    import scipy

    print(
        f"python {sys.version.split()[0]} | camb {camb.__version__} | "
        f"numpy {np.__version__} | scipy {scipy.__version__} | "
        f"matplotlib {matplotlib.__version__} | {sys.platform}",
    )


def main(argv):

    if len(argv) > 1 and argv[1].isdigit():

        # Rate mode has to re-exec, since one process gives one
        # answer no matter how many times it is asked.
        import subprocess

        n = int(argv[1])

        bad = 0

        for i in range(n):

            done = subprocess.run(
                [sys.executable, __file__],
                capture_output=True,
                text=True,
            )

            if done.returncode == 1:
                bad += 1

            print(f"  run {i + 1}/{n}: "
                  f"{'NaN' if done.returncode == 1 else 'finite'}",
                  flush=True)

        print(f"\n{bad}/{n} runs returned a NaN lensing potential")

        return 0

    versions()

    finite = one_run()

    print("\nlensing potential is finite" if finite
          else "\nREPRODUCED: lensing potential is NaN")

    return 0 if finite else 1


if __name__ == "__main__":

    raise SystemExit(main(sys.argv))

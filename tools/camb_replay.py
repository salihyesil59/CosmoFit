"""
Replay a capture from `camb_capture.py` against plain CAMB.

    python tools/camb_replay.py <logfile>

No CosmoFit is imported. Every `CAMBparams` recorded during the test
run is unpickled and handed back to CAMB in the same order, and the
result is checked the same way.

The two outcomes are both worth having:

* The replay reproduces the non-finite result at the same call. Then
  the failure follows the inputs, CAMB is being asked something it
  cannot do, and this file plus the log is a bug report.

* The replay is clean where the capture was not. Then the same inputs
  behave differently inside the test process, so the trigger is
  something about that process -- other libraries, threads, memory --
  and not the inputs. That points back at CosmoFit rather than
  upstream.
"""

import pickle
import sys

import numpy as np
import camb


def load(path):

    records = []

    with open(path, "rb") as fh:

        while True:

            try:
                records.append(pickle.load(fh))

            except EOFError:
                break

    return records


def evaluate(record):
    """
    Reissue one recorded call. Returns (finite, note).
    """

    pars = record["pars"]

    if pars is None:
        # Params that could not be pickled (DarkEnergyPPF's w(a)
        # table). Not replayable, and not counted either way.
        return True, "skipped, unpicklable"

    try:
        results = camb.get_results(pars)

        powers = results.get_cmb_power_spectra(
            pars, CMB_unit="muK", raw_cl=True,
        )

        potential = results.get_lens_potential_cls(lmax=record["lmax"])

        sigma8 = float(results.get_sigma8_0())

    except Exception as exc:  # noqa: BLE001 - diagnostic
        return None, f"{type(exc).__name__}: {exc}"

    finite = (
        bool(np.all(np.isfinite(powers["total"])))
        and bool(np.all(np.isfinite(potential)))
        and bool(np.isfinite(sigma8))
    )

    return finite, f"sigma8={sigma8:.6f}"


def main(path):

    records = load(path)

    print(f"camb {camb.__version__} | replaying {len(records)} calls "
          f"from {path}")

    captured_bad = [
        r["index"] for r in records
        if r["finite"] is False or r["raised"]
    ]

    print(f"the capture itself had {len(captured_bad)} bad calls"
          + (f", first at #{captured_bad[0]}" if captured_bad else ""))
    print()

    replay_bad = []

    for record in records:

        finite, note = evaluate(record)

        if finite is not True:

            replay_bad.append(record["index"])

            if len(replay_bad) == 1:
                print(f"REPLAY REPRODUCED at call #{record['index']}")
                print(f"  model={record['model']} "
                      f"lmax={record['lmax']} "
                      f"acc={record['lens_potential_accuracy']}")
                print(f"  {note}")

        if record["index"] % 50 == 0:
            print(f"  ...{record['index']} replayed", flush=True)

    print()
    print(f"capture: {len(captured_bad)} bad   "
          f"replay: {len(replay_bad)} bad")

    if captured_bad and not replay_bad:
        print("\nVERDICT: the same inputs are fine outside the test "
              "process.\nThe trigger is the process, not what CAMB was "
              "asked. Not an upstream bug on this evidence.")

    elif replay_bad:
        print("\nVERDICT: the failure follows the inputs. Reproducible "
              "against plain CAMB,\nso this log plus this script is a "
              "bug report.")

    else:
        print("\nVERDICT: nothing failed in either. Capture a run that "
              "actually fails.")

    return 0


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise SystemExit(__doc__)

    raise SystemExit(main(sys.argv[1]))

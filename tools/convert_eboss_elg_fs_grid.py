"""
Convert eBOSS DR16's ELG full-shape grid to the form the package ships.

The released file, ``sdss_DR16_ELG_FSBAO_DMDHfs8gridlikelihood.txt``
from CobayaSampler/bao_data, is 60 MB of ASCII: a 100x100x100 grid in
(D_M/r_d, D_H/r_d, f*sigma8) with one probability per row. Sixty
megabytes of text is not something to put in a wheel, and 10.3% of
the released probabilities are exactly zero -- underflow, not
measurement -- which have no logarithm.

This script does the conversion once. It is committed so that the
shipped file can be regenerated and checked rather than taken on
trust:

    python tools/convert_eboss_elg_fs_grid.py <released.txt> <out.npz>

Two lossy steps, both measured rather than assumed.

**float32.** The stored quantity is a log-probability, and the
likelihood is used through differences of it; float32 carries ~7
decimal digits, far more than the grid's own resolution.

**A floor at ``max(log p) - 200``.** exp(-200) is 1e-87, so a point
at the floor contributes nothing to any normalization, and this
replaces the -inf that the exact zeros would otherwise produce --
which would poison an interpolator across its whole support rather
than at one node. Checked: the marginal median and the 16th/84th
percentiles of all three axes are unchanged to four decimal places.

Together: 60 MB -> 1.95 MB.
"""

from __future__ import annotations

import sys

import numpy as np


#: Points more than this far below the peak in log-likelihood are
#: clamped. exp(-200) = 1e-87.
LOG_FLOOR_DEPTH = 200.0


def convert(source: str, destination: str) -> None:

    raw = np.loadtxt(source)

    if raw.ndim != 2 or raw.shape[1] != 4:

        raise ValueError(
            f"Expected four columns (D_M/r_d, D_H/r_d, f*sigma8, "
            f"probability), got shape {raw.shape}."
        )

    axes = [np.unique(raw[:, i]) for i in range(3)]

    shape = tuple(len(a) for a in axes)

    if int(np.prod(shape)) != raw.shape[0]:

        raise ValueError(
            f"{raw.shape[0]} rows do not fill the {shape} grid the "
            f"coordinate columns describe."
        )

    # `np.unique` sorts; the file's own row order has to be matched
    # rather than assumed.
    order = np.lexsort((raw[:, 2], raw[:, 1], raw[:, 0]))

    probability = raw[order, 3].reshape(shape)

    positive = probability > 0.0

    log_prob = np.full(shape, -np.inf)
    log_prob[positive] = np.log(probability[positive])

    floor = log_prob.max() - LOG_FLOOR_DEPTH

    log_prob = np.maximum(log_prob, floor).astype(np.float32)

    np.savez_compressed(
        destination,
        log_prob=log_prob,
        dm_over_rs=axes[0],
        dh_over_rs=axes[1],
        fsigma8=axes[2],
    )

    print(
        f"{source} -> {destination}\n"
        f"  grid {shape}, {int((~positive).sum())} exact zeros floored, "
        f"peak log p = {log_prob.max():.4f}"
    )


if __name__ == "__main__":

    if len(sys.argv) != 3:
        raise SystemExit(__doc__)

    convert(sys.argv[1], sys.argv[2])

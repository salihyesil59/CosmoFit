"""
Which DES-SN5YR release is bundled, and that the citation says so.

The files are the DES-Dovekie recalibration (Popovic et al. 2026),
byte-identical to ``4_DISTANCES_COVMAT/DES-Dovekie_HD.csv`` in
des-science/DES-SN5YR -- 1820 supernovae. Every citation in the
library named the original 2024 release instead (1829 supernovae,
different distances to the same objects, a different w0-wa result).
Nothing in the data was wrong; what a fit with it should be quoted
against was.
"""

from __future__ import annotations

import numpy as np

from CosmoFit.data.loader import dataset_reference, load_des_sn5yr


def test_bundled_sample_is_the_dovekie_one():

    data = load_des_sn5yr()

    assert data.size == 1820
    assert data.covariance.shape == (1820, 1820)

    # 1623 DES supernovae and 197 low-z anchors, from the IDSURVEY
    # column; checked through the redshift range the two cover.
    assert np.min(data.z_hd) > 0.025
    assert np.max(data.z_hd) < 1.15


def test_citation_names_the_release_that_is_bundled():

    reference = dataset_reference("des_sn5yr")

    assert "2511.07517" in reference
    assert "Dovekie" in reference

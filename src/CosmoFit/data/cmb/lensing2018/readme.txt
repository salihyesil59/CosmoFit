Planck 2018 CMB lensing likelihood -- the "conservative" baseline.

Source
------
CobayaSampler/planck_supp_data_and_covmats, lensing/2018/, which
redistributes the Planck Legacy Archive PR3 lensing likelihood. The
original files all carry the prefix

    smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8

which names the reconstruction: the SMICA foreground-cleaned map,
the minimum-variance (MV) TEB estimator, and the *conservative*
multipole range 8 <= L <= 400 that Planck 2018 adopts as baseline.
The prefix is dropped here and the structure kept; nothing else is
changed, and no file is reformatted.

    bandpowers.dat                    <prefix>_bandpowers.dat
    cov.dat                           <prefix>_cov.dat
    lensing_fiducial_correction.dat   <prefix>_lensing_fiducial_correction.dat
    window/window{1..9}.dat           <prefix>_window/window{1..9}.dat
    lens_delta_window/window{1..9}.dat  <prefix>_lens_delta_window/...

Contents
--------
bandpowers.dat   9 bandpowers of [L(L+1)]^2 C_L^phiphi / 2pi over
                 L = 8-400, with their errors.
cov.dat          the 9x9 bandpower covariance.
window/          W_bL, mapping a theory C_L^phiphi to bandpower b.
lens_delta_window/
                 the linear-correction windows M^X_bL for
                 X = TT, EE, TE, PP (in that column order), which
                 propagate the dependence of the reconstruction
                 normalization on the CMB spectra it was measured
                 from.
lensing_fiducial_correction.dat
                 the value of that correction at the fiducial
                 cosmology, subtracted so the correction vanishes
                 there.

The aggressive variant (agr2, L = 8-2048) exists in the same source
directory and is not bundled: Planck's own baseline is the
conservative range, and the aggressive one needs the high-L
reconstruction whose systematics Planck flags as less well
controlled.

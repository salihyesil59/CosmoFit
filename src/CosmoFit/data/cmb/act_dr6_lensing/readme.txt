ACT DR6 CMB lensing likelihood data.

Source
------
ACT_dr6_likelihood_v1.2.tgz from NASA's LAMBDA archive
(https://lambda.gsfc.nasa.gov/product/act/actadv_prod_table.html),
directory v1.2/. Copied unchanged; the tarball is 345 MiB and holds
the whole DR6 likelihood suite, of which these three files (1.3 MB)
are the lensing part.

    clkk_bandpowers_act.txt   18 bandpowers of C_L^kappakappa.
    binning_matrix_act.txt    (18, 3000): row b maps a theory
                              C_L^kappakappa on l = 0..2999 to
                              bandpower b.
    covmat_act_cmbmarg.txt    (18, 18), the *CMB-marginalized*
                              covariance.

Which covariance, and why
-------------------------
ACT ship two. `covmat_act.txt` is the raw one, and using it requires
an explicit correction for the reconstruction's dependence on the
primary CMB spectra it was normalized against -- whose response
matrix is a (n_bin, 4001, 4001) array that cannot be shipped inside
a library.

`covmat_act_cmbmarg.txt`, bundled here, has that dependence already
marginalized over. ACT recommend it when the lensing measurement is
not combined with primary CMB data. Combining anyway is conservative
rather than wrong -- see likelihoods/act_lensing.py.

`covmat_act.txt` is deliberately *not* bundled: nothing here can use
it, and shipping data no code reads is how the low-l TT files sat
unused for three releases.

Variants
--------
The released vector runs wider than the range ACT adopt.

    act_baseline   bins [2:-6]  -> 10 bandpowers,  L = 40-763
    act_extended   bins [2:-3]  -> 13 bandpowers,  L = 40-1250

The ACT+Planck joint variants (actplanck_*) need additional files
and would overlap this library's own Planck lensing dataset; they
are not bundled.

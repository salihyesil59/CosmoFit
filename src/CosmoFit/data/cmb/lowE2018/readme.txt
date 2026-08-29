Planck 2018 low-multipole EE likelihood (SimAll), as a probability
table.

Source
------
CobayaSampler/planck_native_data, release v1, asset
planck_2018_lowE.zip -- a Python translation of the public Planck
clik likelihood
simall_100x143_offlike5_EE_Aplanck_B.clik, with the data converted
to this table. Copied unchanged.

Format
------
A (3000, 28) array of log-probabilities. Column j is multipole
l = j + 2, so l = 2..29. Row i is the value
D_l^EE = l(l+1)C_l^EE/2pi = i * 1e-4 muK^2.

The likelihood is therefore a lookup, not a Gaussian:

    log L = sum_l  table[int(D_l^EE / 1e-4), l - 2]

Why a table and not a mean and an error bar
-------------------------------------------
At l < 30 there are only 2l+1 modes on the sky, so the C_l
distribution is strongly non-Gaussian and skewed -- and this is the
regime that carries essentially all of the CMB's information about
the reionization optical depth tau. Compressing it to
tau = 0.054 +- 0.007 (which is what CosmoFit's "tau" dataset does)
is the standard shortcut and a real approximation. This is the
thing itself.

# Changelog

Newest first. Every version below was published as a GitHub release.
From `v0.25.0` to `v0.40.0` those are `-dev` pre-releases cut from
the `dev` branch, while `main` stayed at `v0.22.0`; **`v1.0.0` is
where `main` catches up with all of it**. Version numbers bumped in flight but never
released are folded into the release that carried them.

The record starts at `v0.3.0` -- `v0.1` and `v0.2.0` predate it. The
entries up to `v0.25.0` were kept in the README's "Project Status"
section and were moved here unchanged.

Each entry says what changed and, where it matters more, *how it was
found out to be wrong* -- a bug that produced a plausible number is
worth more words than a feature that worked first time.

## Unreleased

### `FTPowerLaw`, and the sibling that was missing

The modified-gravity family had f(Q), f(R) and f(R,T) but no f(T) --
the one theory in that group whose action the `theory` module's own
documented example already derives. **`FTPowerLaw`** adds it: the
Bengochea-Ferraro power law `f(T) = T + alpha T^n`, with `T = 6H^2`
and general relativity at `f(T) = T`.

`alpha` is not a free parameter. The `E(0) = 1` closure fixes it from
`Omega_m`, so `n` is the only quantity beyond flat LCDM's `H0` and
`Omega_m` -- the same shape as `FQExponential`, whose `lambda` is
fixed the same way.

Two limits are exact rather than approximate, and both are asserted
as equalities rather than tolerances: `n = 0` is flat LCDM
identically, since `f = T + alpha` is teleparallel GR plus a
cosmological constant; `n = 1` is Einstein-de Sitter whatever
`Omega_m` is, since `f = (1 + alpha) T` is a rescaled TEGR whose
rescaling cancels out of the Friedmann equation. The second is the
one that earns its keep -- it catches a closure constant that
wrongly survives into a case where it must vanish, which closure
alone would not.

Growth: `mu(a) = 1/f_T`, scale-independent, exactly 1 at both GR
limits (`n = 0`, and `Omega_m = 1` at any `n`).

### The pole that was not where it looked

The first implementation carried the closure amplitude
`A = (Omega_m - 1)/(2n - 1)` through the Friedmann solve, and
refused to build at all near `n = 1/2` where that diverges. That was
wrong in an instructive way: the background does not depend on `A`,
it depends on `(2n - 1) A`, and the closure makes that combination
exactly `Omega_m - 1` -- finite everywhere. Written that way the
Friedmann relation is

    E^2 + (Omega_m - 1) * E^(2n) = Omega_m * (1+z)^3

with no free amplitude in it and no pole. `E(z)` and `dEdz` now work
at any `n`, `n = 1/2` included, and only `mu` refuses there, because
only `mu` genuinely diverges: holding the background fixed at
`n = 1/2` forces `alpha` to infinity, which switches gravity off.
A model that had refused to answer a question it could answer
perfectly well.

### Checked against two independent implementations

The Friedmann equation, the closure and `mu` were derived by a
minisuperspace tetrad variation done symbolically in the
[wljs-gr-toolkit](https://github.com/salihyesil59/wljs-gr-toolkit)
GR-06 notebook, and checked there against the TEGR, LCDM and
Einstein-de Sitter limits before any Python was written.

The Python then agrees to ~1e-16 with *two* separate routes: that
Wolfram derivation, and this library's own `theory.Action` reduction
of `T + A0*(-T)**b`. The second is the interesting one. `theory`
uses the opposite sign convention for the torsion scalar,
`T = -6H^2`, so the two started out looking like different models;
working through both showed the closure absorbs the difference
exactly, and the agreement now covers the convention as well as the
algebra. `tests/test_ft.py` keeps both comparisons, the second
skipped when sympy is absent.

The README's action-versus-hand table gains the row this makes
possible: `T + A0*(-T)**b` had only ever been checkable at `b = 0`,
against LCDM, because there was no hand-written f(T) to compare the
general case with.

### `FTPowerLaw` returned plausible numbers where it has no solution

Found by a `RuntimeWarning` in the test log, not by a failing test,
which is the point.

On the `n > 1/2` branch the Friedmann relation's left-hand side
`E^2 + (Omega_m - 1) E^(2n)` is not monotonic: with `Omega_m < 1` it
turns over, so past some redshift there is **no real E at all**. At
`n = 2`, `Omega_m = 0.315` that happens above `z = 0.05`. Newton does
not report this. It wandered off and returned whatever it landed on,
and what it landed on looked entirely reasonable -- finite, and
positive: `E(0.5) = 4.53`, `E(2.0) = 1.70`. Smaller at higher
redshift. An expansion history running backwards, handed back without
a word.

The solve now stays on the physical branch and, more to the point,
**verifies its own answer**: the returned root has to satisfy the
equation it was solving, with a residual scaled by the right-hand
side. Where it does not, the answer is NaN, which is what the rest of
the library returns for a model evaluated where it has none. Nothing
in the supported range changes -- the `E(z)` and `mu` values still
match the Wolfram derivation to 4e-16, and the LCDM limit is still
exact to the last bit.

Three tests were added around it: that the no-solution regime is NaN
rather than a number, that `E(z)` increases with `z` wherever it is
finite (the invariant the unconverged result violated), and that the
solver emits no warning, since a warning is how this hid in the first
place.

### Two CAMB tolerances that were a coin toss

`test_planck_lensing.py` and `test_act_lensing.py` each bump the
primordial amplitude, restore it, and assert the spectrum comes back
to `rtol=1e-12`. CAMB is not bit-reproducible between calls: its
OpenMP reductions depend on how many threads the runtime actually
uses, which varies with machine load. Measured on the lensing
bandpowers, one thread against eight differ by up to **1e-10**
relative -- a hundred times the tolerance being asserted. A captured
failure missed by 1.45e-12.

Both are now `1e-6`: four orders above that noise floor and five
below the effect being checked, which is a ~40% change at the
smallest bandpower. The discriminating power is intact -- a restore
that is wrong by one part in 10^4 still moves the spectrum by
3.7e-5, thirty-seven times the new tolerance.

**What this does not fix**, and what turned out to be behind most of
it, is in the next entry.

### A NaN spectrum could be rejected as if it were bad physics

CAMB can return an all-NaN power spectrum without raising. Nothing
checked, so the NaN became a NaN chi2, and a non-finite chi2 is
turned into `-inf` by `LogPosterior`, which is a rejection. A
sampler therefore dropped points quietly, and the output said
nothing about it.

That conflates two things that are not the same. Rejecting a point
because the parameters are unphysical is the model doing its job.
Rejecting one because the solver failed at parameters that are
perfectly reasonable is a hole in the sampled volume, and it biases
the result in a direction nothing else records. The failures seen
here were at Planck's own best fit -- `H0 = 67.36`,
`Omega_m = 0.3153`, `ln1e10As = 3.044`, `tau = 0.0544` -- so "bad
parameters" was never the explanation.

Two changes, on the two sides of the problem.

**`CAMBBackend` no longer hands out a result it cannot vouch for.**
Every returned quantity is checked, `sigma8` included, and a
non-finite one raises `BoltzmannError` naming both the fields that
went bad and the point in parameter space, so the report is
something you can act on.

**`LogPosterior` separates a solver failure from an unphysical
point.** The point is still rejected -- a chain cannot stop because
CAMB had a bad moment -- but it is counted in `solver_failures`,
the first one is kept in `first_solver_failure`, a warning goes out
the first time it happens, and `run_mcmc` reports the total when it
finishes. A handful in a long run is noise; a large fraction means
the result should not be trusted until the cause is found. Either
way the number is now readable instead of invisible.

There is no retry. One was written, and then measured: an identical
repeated call returns NaN again, so the failure is a property of
the process rather than a transient glitch, and retrying only
doubled the cost of the most expensive call in the library.

**What is still unknown, and what it means for CI.** Why CAMB
returns NaN is not established, and the evidence points outside this
library. Ruled out by direct test: the parameter points (every
reported one computes cleanly in a fresh process), a single
triggering module (each CAMB-using module paired with the first
failing one passes), memory (the failure happened at 907 MB with
neighbouring tests passing at the same figure), transience (see
above), and any special role for small `lmax` (a sweep from 2 to
100 is clean). What is left is CAMB's own state after many calls in
one process.

The consequence is worth stating plainly: **this makes the test
suite fail when it happens**, where before it sometimes passed with
a silent NaN. Runs of 0, 1, 10 and 13 failures have all been seen.
That is the intended trade -- a suite that goes red on a real
problem is better than one that stays green by not looking -- but
it does mean a red run here may be this, and not the change in
front of you.

### One failing CMB test was manufacturing a dozen more

The Planck modules had looked unstable for a while: consecutive
full-suite runs produced 0, 1, 11 and 14 failures across
`test_planck_lensing`, `test_planck_lite` and `test_planck_lowe`,
while each of those modules passed when run alone. That reads like
flakiness. Most of it was a cascade.

`test_planck_lensing`, `test_planck_lowe` and `test_act_lensing`
each build their likelihood in a **module-scoped** fixture, so every
test in the file shares one mutable cosmology, and several of them
move a parameter, check what it does, and move it back. That works
until one of them fails -- at which point the restore line never
runs, and the cosmology stays moved for everything after it.

Caught in one run with full tracebacks. CAMB returned NaN for the
baseline spectrum; the first two tests failed on that; the third
failed before undoing its `ln1e10As += 0.10`; the fourth failed
before undoing its own `+= 0.20` and `n_s -= 0.03`; and by the
sigma8 check the shared cosmology was 0.30 high in `ln1e10As` and
reported `sigma8 = 0.932` against an expected 0.811. Thirteen
failures, one cause. How far the cascade travelled decided the
count, which is exactly why it looked random.

A new `tests/conftest.py` restores the shared cosmology after every
test that uses one, pass or fail, and is a no-op for the rest of the
suite. Verified against a deliberately failing test: without it the
next test fails too, with it the next test sees the cosmology it
expects.

**Still open.** This does not explain why CAMB returned NaN for a
standard LCDM in the first place, and that remains unfixed -- it did
not reappear in the runs after this change, but three clean runs are
not proof for something intermittent. What the fix does buy is that
the next time it happens the suite reports the one real failure
instead of thirteen invented ones, which is the difference between a
diagnosable bug and noise. Both the NaN and the module-scoped
fixtures predate the f(T) work; the cascade reproduces on the commit
before it.

**Since traced, and it is not ours** -- see the next entry.

### The NaN spectrum, traced out of this library entirely

The intermittent all-NaN spectrum above is reproducible in about six
seconds by a script that imports nothing from CosmoFit. Three
ingredients, all of them required, any two harmless:

    import scipy.stats
    font.set_text("Redshift z")      # one FreeType layout
    camb.get_results(pars)           # lensing potential is NaN

`tools/camb_nan_repro.py` is that script; it reports 7 runs in 10 on
this machine. Each run needs a fresh process, because the outcome is
decided once per process -- repeating the call inside one process
gives the same answer every time, which is why the guard's message
says so.

**What breaks is narrow.** In the same call, the background, the
transfer functions, `P(k)`, `sigma8` and the *unlensed* CMB spectra
are all correct. Only `lens_potential` is NaN, and `lensed_scalar`
and `total` inherit it. `NonLinear = none` makes the same call
finite, so it is confined to the non-linear lensing branch.

**How it was narrowed**, by rates over >= 8 fresh processes each,
never single runs -- the failure is a coin flip and bisecting it one
run at a time produces noise:

| in the process | NaN |
| --- | --- |
| a fitter, and a 1200-evaluation chain | 0/8 |
| a figure built, with labels, legend and styling | 0/10 |
| `tight_layout()` on that figure | 9/10 |
| the same, with no CosmoFit imported | 0/8 |
| the same bbox measurement, no text on the figure | 0/8 |
| `scipy.stats` + one `FT2Font.set_text` | 4/8 |
| `linalg`, `special`, `integrate`, `interpolate`, `optimize`, `sparse` | 0/8 each |

Four explanations were tested and are wrong. It is **not the
floating-point mode**: MXCSR and the CRT control word are identical
(`0x1fa0`) in 8 of 8 failing runs. It is **not memory pressure**:
substituting small numpy churn, large numpy churn or Python object
churn for the text layout is 0/8 in all three. It is **not scipy's
bundled OpenBLAS**: `scipy.linalg` is what loads that, and alone it
is clean. And it is **not HMcode's physics**: its feedback output is
byte-identical between failing and clean runs, and the non-linear
`P(k)` it produces is finite.

Two earlier conclusions in this file are superseded by that. The
trigger was recorded as CosmoFit's plotting path; the plotting path
was simply the first thing in the suite that laid out text. And the
failure was recorded as not reportable upstream, on the grounds that
replaying the captured `CAMBparams` outside the test process came
back clean -- true of the inputs, but too wide a conclusion, since
the reproduction turns out to need no cosmology code at all.

**Nothing here is a correctness risk for a fit.** Everything upstream
of the lensing step is bit-correct in a failing run, the guard
refuses to hand a NaN spectrum to a likelihood, and the posterior
counts these separately from ordinary rejections. The remaining work
was a report to the upstream projects, not a change here; that is
filed as [cmbant/CAMB#210](https://github.com/cmbant/CAMB/issues/210).

One more measurement went into that report and is worth keeping here:
the failure is **order-dependent**. Importing `scipy.stats` *after*
the glyph layout rather than before it is clean, 0/8, so what decides
it is which initialises first and not merely that both are loaded.
No thread setting is involved either -- with `OMP_NUM_THREADS` unset
the reproduction is 8/8 on 16 cores.

### `f(R)` growth, which was the gap

`theory.Action` could solve a general `f(R)` background from the
action but refused `growth="quasi_static"`, because f(R)'s coupling
is scale-dependent -- the scalaron has a Compton wavelength -- and
the scale-free `mu = 1/f'` the teleparallel sectors use would have
been wrong rather than approximate. That refusal was correct and is
now unnecessary:

    mu = (1/f_R) (1 + 4m)/(1 + 3m),   m = (k/a)^2 f_RR/f_R

with `R` read off the model's own integrated background, in
`theory.curvature.quasi_static_mu`. A brand-new f(R) can now be
fitted to growth data, not only to distances -- which matters more
than it sounds, because the background of a modified-gravity model
can usually be tuned to imitate LCDM while its growth cannot.

**The algebra was not the risk.** The GR-05 notebook in the
companion toolkit derives `G_eff` from the perturbed field
equations; subtracting its result from the expression above gives
identically zero in Wolfram. That was checked against the
derivation, not against the notebook's summary of it.

**The units were the risk.** `m` is dimensionless only if `(k/a)^2`
and `f_RR` are in matching units, and misplacing a factor of `c`,
of `100`, or of `a` moves the Compton wavelength without making
anything raise -- growth would come out smooth, plausible and
wrong. So the load-bearing test holds the new formula against
`FRHuSawicki`, which performs the same conversion through
independently written code, and gets agreement to 2e-16 across
three scale factors and three wavenumbers. The remaining tests
pin the two physical limits (`1/f_R` far outside the Compton
wavelength, `4/(3 f_R)` far inside), monotonicity in `k` between
them, and that the departure from GR falls linearly in the
coupling.

That last one is stated as a *rate* rather than a tolerance, and
the first attempt at it was wrong for an instructive reason:
"mu is 1 when alpha is small" is not well posed, since for any
alpha there is a `k` where `m` is not small. What is true at fixed
`k` is that the departure falls linearly in alpha, and the test now
asserts that.

Two things this is not. It is the **linear** result -- chameleon
screening is non-linear and absent, so where screening matters this
overstates the departure from GR. And a metric action *linear* in R
still declines `quasi_static`, which is right: that theory is
general relativity, where `mu = 1` exactly and `growth="gr"` is the
answer rather than an approximation. The error message used to
misdescribe both cases and now says so.

### The other direction, so the interesting f(R) models can be fitted

The previous entry ends by saying the fix is forward integration or a
designer construction. This is the first of those.

`Action(..., background="forward", closure="Lam")` integrates from
deep in matter domination towards today instead of away from it.
Backwards, the scalaron's oscillating mode grows and the problem is
ill conditioned; forwards it decays, so a slightly wrong initial
condition relaxes onto the attractor rather than leaving it. The
arctan model that could not be integrated at all now runs cleanly:
`E(0) = 1` to 1e-12, `R` rising monotonically, `f_R -> 1` and
`f_RR -> 0` at high curvature, `fsigma8` computed.

**Two things swap over.** `R_0` stops being a parameter and becomes
derived -- on the attractor the curvature today is not free, and
offering it as an input was always slightly dishonest. And a
`closure` becomes required, because `E(0) = 1` is now the condition
to satisfy rather than the point one starts from. That restores the
same arrangement second-order actions have.

**The direction is not universally better** and is not chosen
automatically. `R + alpha R^2` wants the backward path, where its
mode decays; the disappearing-cosmological-constant family wants
this one. Which a theory needs is a property of the theory, so it is
an argument rather than a guess.

**The assumption, and how it is checked.** Forward integration
cannot start deep enough for the growth ODE, which wants `E(z)` to
`z ~ 10^4`: the scalaron oscillates too fast there to follow. So
below the starting redshift the history is continued analytically,
on the grounds that this family has returned to General Relativity
and matter dominates by five orders of magnitude. That assumption
does real work, so it gets two tests. Moving the junction from
`z = 20` to `z = 60` must not move `Lam*`, `E(z)` or `fsigma8` --
and does not. And a model still 1.8% away from matter domination
where it starts is refused rather than spliced.

Three mistakes made writing those tests are worth recording, since
each would have left a green test proving nothing:

* `z_init=_Z_INIT` as a *default argument* freezes the starting
  redshift at import, so the test that moves it moved nothing and
  reported a difference of exactly zero -- a run compared against
  itself. Read at call time now.
* The first transient measure compared `R` to a straight line.
  `R` goes as `(1+z)^3`, so that number is dominated by ordinary
  curvature and came out *larger* near today. It now measures the
  largest relative *drop*, which is identically zero for a monotone
  curve and therefore measures the oscillation and nothing else.
* With that fixed the transient is 4.3% near the start, not the
  "part in a thousand" a coarser sampling had suggested and which
  had already been written into a docstring. Both the bound and the
  claim were corrected; loosening the bound alone would have left
  the prose lying.

The transient is tolerable because it is confined: exactly zero
drop over `z < 10`, which is the range anything is fitted against.

This module is the slowest in the suite at around twenty minutes,
since every model has its closure shot for. Noted rather than fixed.

### Two gaps in the action namespace, and one in what it can integrate

Trying to compile a published arctan `f(R)` found three things, in
increasing order of interest.

**`atan` was not in the namespace**, nor any inverse trigonometric or
inverse hyperbolic function. The arctan f(R) models
(arXiv:1601.07928, arXiv:1310.6915) and the arcsin one
(arXiv:1507.04927) cannot be written without them, so the library was
declining a published family with a name error rather than with a
physics answer. Added.

**`pi` was not either.** A bounded correction has to be normalised by
its own saturation value, and an arctan saturates at `pi/2`; writing
`3.14159...` instead would be uglier and inexact. Added as the only
constant -- sympy's other one-letter ones, `E` and `S` especially,
read as parameter names and would shadow them silently, which is the
failure the namespace check exists to prevent.

**And then it still would not integrate**, which is the finding worth
keeping. `theory.curvature` integrates backwards from today, and that
is well conditioned only while the scalaron's oscillating mode does
not grow into the past -- true when `f_RR` is roughly constant, as in
`R + alpha R^2`, and false for the whole "disappearing cosmological
constant" family where `f_RR` falls steeply with `R`.

Measured, on `R - (4 Lam/pi) atan(R/Rw)` with `Lam` tuned so `dR/dN`
at `N = 0` matches an LCDM background exactly: usable only to
`z ~ 1.2`, after which `R` turns over. Not the solver's fault -- RK45,
DOP853, Radau and BDF all fail at the same point, and LSODA reports
success while returning a non-monotonic `R` and NaNs past `z ~ 2.4`,
which is worse than failing. Not a slightly wrong initial condition
either: scanning `R_0` from 8 to 15 never reaches beyond `z ~ 1.2`,
since the precision needed in `R_0` grows exponentially with the
redshift wanted.

The fix is not a better integrator. It is to integrate forwards from
deep in matter domination, where the attractor attracts, or to impose
the background and solve for the `f` that produces it -- the designer
construction `FRHuSawicki` uses, which is why that model is written by
hand rather than compiled from an action. Documented in
`theory/curvature.py` so the next person meets the limit as a stated
scope rather than as a mysterious failure.

### An admissibility gate for `f(R)`

Inventing an `f(R)` that fits data is easy and, on its own, worth
little: a free function can be tuned to almost anything. What
separates a proposal from a curve is whether its perturbations
behave, so a compiled `f(R)` now checks the two standard conditions
along its own background -- `model.viability()`, with
`model.scalaron(z)` returning `(f_R, f_RR)` directly.

`f_R > 0` keeps the graviton from being a ghost. `f_RR >= 0` keeps
the scalaron from being tachyonic -- the Dolgov-Kawasaki
instability, whose growth time is short enough that such a
background would not survive to be observed. Zero is deliberately
*not* flagged: that is the general-relativity limit, where the
scalaron is absent rather than sick, and flagging it would reject
the one limit every f(R) has to pass through.

The reason to automate this is that a non-viable f(R) does not
announce itself. Flipping the sign of `alpha` in `R - 2L + alpha R^2`
leaves `E(z)` smooth, finite and entirely plausible; nothing raises
and the model fits happily. What actually breaks is `mu`: with
`f_RR < 0` the denominator `1 + 3m` passes through zero, and past
that pole the formula still returns a finite number. It no longer
does -- `quasi_static_mu` refuses and names the condition, on the
same principle as the Boltzmann guard.

Not in this: the *mode count*. Confirming from the Hamiltonian
analysis that a general f(R) propagates exactly one extra scalar is
a job for the GR-09 notebook, and it carries a specific trap worth
recording before anyone attempts it. In ADM variables
`R = K_ij K^ij - K^2 + (3)R` **plus total derivatives**. Those are
droppable in general relativity, where they multiply a constant;
in f(R) they multiply `f'(chi)` and are not. Dropping them yields a
plausible and wrong count -- the same trap `theory.curvature`
already documents for the ordinary reduction, which is why that
module uses a Lagrange multiplier instead.

### Checked against the notebooks that derive the same physics

The `wljs-gr-toolkit` notebooks reduce a gravitational action on FLRW
in Wolfram Language; :mod:`theory.minisuperspace` does the same
reduction in sympy. The two share a convention list and nothing else
-- different languages, different symbolic engines, different root
finders, written years apart -- and they had never been compared.

They agree to **3 parts in 1e16**, machine precision, for LCDM and for
the f(T) power law at `n = -0.5, 0.2, 0.7`, through both routes this
library offers: the hand-written classes in `cosmology.models` and the
action compiler in `theory.Action`. `tests/test_notebook_agreement.py`
pins the references, which the toolkit's `cosmofit-reference.wls`
regenerates by loading its GR-02 notebook and driving that notebook's
own functions rather than re-deriving anything.

The algebra matching is the stronger half. GR-02's f(T) constraint is

    E^2 - 6^(n-1) alpha (2n-1) H0^(2n-2) E^(2n) = Omega_m (1+z)^3

and imposing `E(0) = 1` sends the coefficient to `1 - Omega_m`, which
is exactly the relation `FTPowerLaw` was written to solve. Two
implementations arriving at the same *equation* before they arrive at
the same numbers is what makes the numerical agreement mean something
rather than being two solvers agreeing about one shared mistake.

`tests/test_ft.py` already pinned Wolfram values for one model from the
GR-06 notebook, so this extends an existing practice rather than
inventing one; what is new is covering GR-02's engine and the action
compiler.

One incidental result worth recording: GR-02 gives the f(T) and f(Q)
power laws the **identical** background constraint, because both
scalars are `-6H^2` on flat FLRW. That is correct rather than a bug,
and it means background data alone cannot separate those two families
-- which is an argument for the growth observables, not against the
models.

### Carried forward, not hidden

The same toolkit's GR-08 notebook finds that f(T)'s extra Lorentz
modes have kinetic terms that vanish identically around flat FLRW.
They do not propagate at linear order there, which is strong
coupling, and it means linear perturbation theory is not a reliable
guide to them. `mu` here is a statement about the metric sector,
which is well behaved -- it is not a claim that the full theory is.
Gravitational waves in f(T) are exactly luminal, so GW170817 does
not constrain it. Both points are in the class docstring and in
`REFERENCES.md` rather than left for a user to discover.

## v1.0.0

The first stable release, and the point where `main` catches up with
sixty-four commits of `dev`.

What makes it 1.0.0 is not a feature. It is that the three things the
Roadmap had listed since the beginning -- a stable public API,
complete documentation, test coverage across the whole library --
are done and *held there by tests* rather than by intention. The
rest of this entry is the last of that work.

### The GUI catches up with three releases

`CosmoFit.theory` landed in `v0.36.0` and the GUI had no route to it.
It offered one way to a model the library does not ship -- `Custom`,
which takes an `E(z)`, the *result* of a derivation somebody did by
hand. **Now it takes the input instead**: a gravitational Lagrangian,
from which the library derives the Friedmann equation and shows it.
Six worked examples to start from, covering every route the module
has -- algebraic, root-solved, Lagrange-multiplier and integrated --
and each a real model rather than a syntax demonstration.

A new **Inference** tab, for four things the library grew in
`v0.36.0` and the app had none of: profile likelihoods (reported as
the Δχ² = 1 crossings, which is what a profile actually measures),
Fisher matrices *with the ratio to the chain* (the comparison is the
point -- it is a Gaussian approximation, and that is the only way to
know whether it holds here), Bayesian evidence by nested sampling,
and tension by two definitions that disagree exactly when the
Gaussian one has stopped meaning anything.

Two smaller gaps: the `eboss_surface` figure, which is what shows why
a Gaussian summary of the two tabulated eBOSS datasets would be
wrong; and `best_fit(restarts=)`, which exists because ΛsCDM once fit
*worse* than the ΛCDM it contains.

And two ways the GUI could hand somebody a result that looks like a
measurement and is not. **ADE derives `Omega_m`** and the parameter
table ticked it anyway, so choosing that model put the app straight
into the state `Fitter` warns about -- a posterior that is the prior
back again. **And the warning itself was invisible**: `Fitter` warns
through `warnings`, which in a Streamlit app means stderr, which
means nowhere. The run loop catches them and renders them now, so
every guard the library grows later shows up without the GUI
restating any of it.

### The package is typed

`py.typed` ships. That tells every downstream type checker to trust
these annotations, so it had to be true first: **201 of 201** public
callables fully annotated, from 58. `CosmoFit.typing` spells out the
two aliases the whole surface is written in -- `E(0.5)` returns
`np.float64` and `E([0.1, 0.2])` returns an `ndarray`, so annotating
the return as `np.ndarray` alone would have been wrong for the
commonest call in the library.

Two tests hold it there: every annotation resolves, *including* the
ones deferred to `TYPE_CHECKING` (which is what a checker sees and
`get_type_hints` cannot -- `matplotlib.figure` costs 0.65 s to
import, more than the rest of the library, and nobody who never draws
a figure should pay it), and nothing on the public surface is left
unannotated.

Running mypy over it for the first time found four annotations that
were simply **false**: `load_chain` declared `MCMCResult` and returns
a `StoredSampler`, `theory.fields.build_system` declared one object
and returns two, `BaseLikelihood.model` declared an array where
`S8Likelihood` returns a number, and `EnsembleSampler.run` narrows
its base class's `**kwargs` into named parameters. All corrected --
which is exactly the class of thing `py.typed` would otherwise have
started telling other people's type checkers.

mypy reports 81 remaining errors, all inside function bodies rather
than in the signatures `py.typed` is a promise about. It is in the
`dev` extra so a contributor can run it; CI does not gate on it yet.

### Two workflows that nothing can trigger by accident

`publish` builds, runs `twine check --strict`, and uploads to PyPI
through Trusted Publishing -- GitHub mints a short-lived OIDC token
and PyPI accepts it in place of an API token, so no long-lived secret
lives in this repository at all. `pages` deploys the API reference to
GitHub Pages.

Both are **`workflow_dispatch` only**, and that is the design rather
than a placeholder. A version number, once published, can never be
reused; a site, once deployed, is on the public internet under this
repository's name. Neither should be able to happen as a side effect
of pushing a tag. Someone has to open the workflow and run it, and
each needs one thing set up on the other side first -- a Trusted
Publisher on PyPI, Pages enabled here.

### The release, rehearsed

Everything that could be checked before a first publish was, rather
than being left to find out on the day:

* `cosmofit` is free on PyPI and TestPyPI.
* Both artifacts pass `twine check --strict`. The wheel is 22 MB
  against PyPI's 100 MB per-file limit -- 10 MB of it the Pantheon+
  covariance -- so no limit increase is needed.
* **The sdist was incomplete**, and nothing would have said so. The
  bundled data travels because it lives inside the package, but
  `CITATION.cff`, `REFERENCES.md`, `CHANGELOG.md` and
  `CONTRIBUTING.md` sit at the repository root and were simply
  absent. A `MANIFEST.in` adds them and prunes `docs/`, `examples/`,
  `app/` and `tools/`, which have no business in a source
  distribution. The citation file is the one that matters: this
  library bundles other people's data and implements other people's
  methods, and the file saying how to credit them should travel with
  the source rather than living only on a web page.
* **The wheel was installed into a clean virtual environment**,
  outside this source tree, and fitted ΛCDM to
  CC+DESI+Pantheon++Planck -- 1671 points, H₀ = 67.5, Ω_m = 0.313 --
  with `py.typed` present and the `theory` extra correctly absent and
  correctly explaining itself.

That last check earned its place immediately: it found a **fifth
false annotation**, and then a sixth. `Fitter.best_fit` was declared
`-> BestFitResult` and returns scipy's `OptimizeResult`;
`Fitter.run_mcmc` was declared `-> MCMCResult` and returns emcee's
`EnsembleSampler`. Both were wrong the same way -- named after the
object that lands on `fitter.result` rather than the one handed back
-- and neither mypy nor the resolution test could see it, because
scipy and emcee both return `Any`.

So there is now a test that **calls** ten of `Fitter`'s methods and
compares what comes back against what was declared. With `py.typed`
shipped, a wrong annotation is no longer a private mistake; it is
something other people's type checkers repeat with confidence.

GitHub Pages is enabled with Actions as the source, which fixes the
documentation URL, so `[project.urls]` gains a `Documentation` entry.
Run `pages` before `publish` and it is live rather than a 404.

### `main` catches up

Sixty-four commits, and the reason this is 1.0.0 rather than 0.41.
`main` had been deliberately frozen at `v0.22.0` since August while
everything went out as `-dev` pre-releases; it now carries all of it.

That freeze had one consequence nobody had noticed: a
`workflow_dispatch` workflow is only visible to GitHub if the file
exists on the **default branch**. `tests` worked because it is
push-triggered, but `publish` and `pages` returned 404 to every
attempt to run them. They could not have been run at all until
`main` moved -- so the last item on the Roadmap turned out to be a
prerequisite for the two before it rather than a tidy-up after them.

The `Changelog` and `Examples` entries in `[project.urls]` move from
`dev` to `main` with it, as does the "edit this page" link in the
API reference. `Development Status` goes from `4 - Beta` to
`5 - Production/Stable`.

And with `main` moved, both workflows could finally run:

* the API reference is live at
  <https://salihyesil59.github.io/CosmoFit/>;
* `pip install cosmofit` works --
  <https://pypi.org/project/cosmofit/>.

Published to TestPyPI first, installed back down from that index into
a clean environment, and checked to give the same fit to the digit
before the real one was touched.

### And the notebooks stop cloning the repository

All seventeen used to open in Colab by cloning `dev` and installing
it *editable*, which then needed a `sys.path` fix: pip runs in a
subprocess, so the already-running kernel never saw the `.pth` file
it wrote, and `import CosmoFit` could resolve to the bare checkout
directory instead -- same name, and `/content` is on `sys.path`. Six
slightly different versions of that workaround had accumulated.

A published package removes all of it: `pip install cosmofit`, into
site-packages, which the kernel is already looking at. Each notebook
asks for the extras it actually needs, and the Colab badges point at
`main` rather than `dev`.

Two things fell out of checking rather than assuming. `lscdm_mcmc`
looked like it needed CAMB because it mentions `planck_lite` -- in a
section explaining why the full spectra **cannot** be used there.
And `dataset_zoo` genuinely does need it, which the extras table in
`examples/README.md` had not said. `dark_energy_evidence_audit` was
the one notebook with no Colab badge at all, while that README
claimed every one of them was Colab-ready; it has one now.

## v0.40.0

Everything the Roadmap listed for **v1.0.0**, worked through against
what the repository actually measured rather than against
recollection. No new physics: this is the release where the library
stops being a research codebase that happens to be public.

### The release history had stopped at v0.25.0

Fourteen releases -- the whole of `CosmoFit.theory`, Bayesian
evidence, profile likelihoods, Fisher matrices, tension statistics,
the three holographic models, the performance pass, the examples
reorganization -- existed only as GitHub release notes. The
repository did not record its own last month of work.

That history is now this file, and the README lost half its bytes
(98 KB to 49 KB) getting it out of the way: the old "Project Status"
section had grown to nearly a thousand lines, so a reader looking for
"what is this library" scrolled through a year of release notes to
reach the Roadmap.

### A public API that is pinned

Nothing was holding it. Every other test imports what it happens to
need, so a name could be renamed, moved between subpackages or
dropped from `__all__` and the suite would go on passing as long as
*some* path to the object still existed.
`tests/test_public_api.py` types the surface out by hand -- 64 names
-- and asserts the set in both directions. An extra name is a promise
nobody meant to make; taking it back out is then itself a break.

Writing it down found two names that had already drifted out of
`CosmoFit.cosmology` while every one of their siblings was
re-exported: **`ModelConfigurationError`**, which is the one
exception a user is asked to tell apart from an ordinary failure and
was reachable only as
`from CosmoFit.cosmology.core.errors import ...`, and
`GrowthCalculator`. Both are importable from the top level now, and
a further test asserts that each layer of `cosmology` is re-exported
*whole*, so the class of drift is closed rather than the two
instances of it.

Two more things the file checks. **Importing the library must not
import an optional backend** -- which caught numba: `kernels.py`
imported it at module scope, so `import CosmoFit` paid 115 ms, 17% of
the whole import, for a kernel only a growth fit ever calls. It is a
spec lookup now and the compilation happens on first call; 0.66 s to
0.56 s, with the two stepping paths still agreeing to 1e-13. And
**every annotation on the public surface resolves** -- these modules
use `from __future__ import annotations`, so a wrong annotation is
invisible until something calls `get_type_hints`, which nothing did.

### Test coverage, where the measurement said to look

93% overall, from 79%, across **681 tests**. The Roadmap's
phrasing -- "across the whole library, not only the newest parts" --
turned out to be exactly right: `theory`, the newest subpackage, was
at 86-94% while the oldest code was not.

| | was | now |
|---|---|---|
| `plots/plotter.py` | 16% | 94% |
| `stats/cpl_diagnostics.py` | 37% | 100% |
| `stats/results.py` | 58% | 97% |
| `stats/chains.py` | 64% | 92% |
| `cosmology/core/parameters.py` | 67% | 96% |
| `likelihoods/joint.py` | 70% | 100% |
| `stats/priors.py` | 73% | 100% |
| `data/covariance.py` | 78% | 93% |
| `stats/posterior.py` | 85% | 100% |

**The plotter is the largest module in the library and 16% of it had
ever run.** Nothing anywhere called a single plotting method. It is
also the worst place for a gap, because a plot fails *quietly* -- an
empty axis, a curve at the wrong scale, a band that collapsed to a
line. Nothing raises; the figure is saved and somebody looks at it.
So the assertions are about content: axis labels set, artists
carrying points, one trace panel per free parameter with one line per
walker, six axes for the three CMB spectra and their residual panels.
Three of them are not about drawing at all -- that a missing dataset
names itself instead of raising `AttributeError` on a `None`, that
the band appears only once there is a chain, and that drawing a band
**leaves the cosmology where it found it** (a band evaluates hundreds
of posterior draws; without the restore, every number read off the
fitter after a plot would come from whichever draw happened to be
last).

Elsewhere the untested half was consistently the *rejection* paths
and the code that leaves the process: the chain-signature machinery
that decides whether a saved chain may be resumed (the quietest
scientific error the library can make -- one array of samples drawn
from two different posteriors, with nothing to say so); the JSON
encoder standing between numpy and a `json` module that rejects
everything numpy produces; both guards on `LogPosterior.chi2`, which
exist because L-BFGS-B once evaluated the objective at
`[nan, nan, nan]`; and both paths through `DenseCovariance.solve`,
including the fallback that discards a precomputed inverse which
fails its own accuracy check.

The sharpest test of the batch is in `cpl_diagnostics`: **the
Mahalanobis distance is not the significance.** `D^2` follows
chi-square with two degrees of freedom, so D = 2.20 is 1.70 sigma,
not 2.2 -- and reporting `distance` as sigma overstates the tension
with LCDM, which is the direction a dark-energy paper would like it
to be wrong in.

### An API reference, and ten docstrings that rendered wrong

`pyproject.toml` had declared a `docs` extra -- sphinx and furo --
for a repository with no `docs/` directory. There is one now:
autosummary over the public API, napoleon for the numpydoc
docstrings, intersphinx, and myst-parser so README.md and
CHANGELOG.md are *included verbatim* rather than copied into a
second, drifting version.

Building it produced **237 warnings**, and they were not cosmetic:

* indented equations are reStructuredText *block quotes*, and a
  continuation line beginning with `*`, `+` or `-` becomes a bullet
  list inside one -- so GEDE's `Omega_de(z)`, IDE's continuity
  equations, GCG's density evolution and the curvature branches of
  `DistanceCalculator.DM` were all being parsed as prose with lists
  in the middle;
* `GCG`, `FRTLinear` and `FRHuSawicki` each had a numpydoc
  `Parameters` section whose body is a *paragraph*, which napoleon
  renders as a field list that the prose then breaks;
* `|beta|`, `|p/rho|` and `|a - b|` -- absolute-value bars in prose
  -- are reStructuredText *substitution references*, and each was an
  error for an undefined name.

The build is clean under `-W` now, and CI keeps it there.

### Packaging, and three CI jobs

`[project.urls]` did not exist, so a PyPI page would have been a
rendered README and a dead end. Four classifiers that matter for a
scientific package were missing, including both
`Scientific/Engineering` topics -- which is how somebody browsing for
a cosmology library would ever find this one -- and 3.12/3.13, which
CI had been testing for months. The license moved to an SPDX
expression.

`black` came *out* of the `dev` extra. It was listed and nothing used
it, which is worse than it sounds: a contributor taking the extra at
its word would reformat 101 of the 117 files under `src/` and
`tests/`, and this code is hand-wrapped. `ruff` is configured as a
linter only, with three rules ignored and a reason for each.

New CI jobs: **lint** (which found seven dead or shadowed names the
test suite could not see, and later refused a blind
`pytest.raises(Exception)` of mine), **`python -m build` +
`twine check`** -- the half of a release that cannot be found by
running the code -- and **docs**, building with warnings as errors.

### Also

`CONTRIBUTING.md` and `CITATION.cff`. The first explains why there is
no formatter and what a finished change looks like here; the second
is what GitHub's "Cite this repository" button reads, and it says
plainly that the papers behind whatever you used matter more than the
software entry.

## v0.39.2

No new capability. This closes a gap in what was *tested*, corrects
two error messages that had fallen behind, and walks back a Roadmap
claim the previous release had itself added. **501 tests.**

### The two things a runtime-built model cannot do

Three routes now produce a `Cosmology` subclass that exists only in
the session that made it -- `define_model` (an `E(z)` function),
`model_from_expression` (an `E(z)` string) and
`CosmoFit.theory.Action.build` (an action). Everything about fitting
works on them, with two exceptions, and both follow from one fact:
such a class cannot be pickled *by reference*, because there is no
importable name to pickle it to. `run_mcmc(n_processes>1)` sends the
model to worker processes; `Fitter.from_chain` re-imports the model
the chain was sampled with.

Both were already guarded, and guarded well -- `run_mcmc` refuses up
front rather than letting the pickling failure surface from inside a
worker, where the traceback says nothing about the model. But neither
had a test, and neither message mentioned `Action.build()`, so
somebody who had written an action was told about two functions they
had never called.

Messages corrected, and tests added across all three routes. The one
worth more than the error text: **the `from_chain` workaround actually
works** -- rebuilt the same way, the chain reloads and gives back the
same posterior to 1e-12. A message promising a fix that does not work
is worse than no message.

### A Roadmap entry walked back

`v0.39.1` rewrote the Roadmap, and one of the entries it *added*
claimed more than it should have. It said the models' own `E(z)`
counts massive neutrinos inside `Omega_m` as pure matter -- true as
far as it goes. But the three places `E(z)` is consumed above z ~ 100
are each already handled: the sound horizon integrates the exact
Fermi-Dirac density (and validates against CAMB's `rdrag` to 5e-5);
the compressed Planck priors deliberately keep massive neutrinos
inside `Omega_m` at every redshift, because a prediction has to share
the compression's definitions rather than improve on them; and the
growth ODE solves a matter-plus-dark-energy equation, so starting it
deep in *that* matter era is self-consistent. The item is real but far
narrower than it read, and doing it carelessly would break agreements
the library has already validated. The Roadmap now says so, rather
than leaving a trap for whoever picks it up.

## v0.39.1

Neither commit adds a model: the first closes two ways a theory-built
model could give a wrong answer with nothing said, and the second
catches the documentation up with four releases of work. 491 tests.

### Growth of structure, where a theory model was silent

Two failures of the same kind -- the answer is wrong, and nothing says
so.

**A configuration error was being swallowed into infinity.**
`LogPosterior.chi2` turns exceptions into an infinite chi-squared,
which is right for a *parameter* the model cannot represent: a sampler
that merely proposed it should not crash, and treating the point as
the worst possible fit is exactly what excluding it means. It is wrong
for a *configuration* it cannot represent. The growth ODE starts at
z = 9999 and a field action's history starts at `z_init`, default
3000 -- so `Fitter(model=<any field model>, datasets=["fsigma8"])`
built fine and then returned `chi2 = inf` at *every* parameter value,
a fit that ran to completion having learned nothing and saying
nothing. The message existed and already named the fix; it just never
reached anyone. New `ModelConfigurationError`, deriving from
`Exception` rather than from any of the three caught there, so it
propagates -- and it now names the growth ODE's starting redshift,
which is the number needed to pick `z_init`.

**Scalar-tensor gravity reported `mu = 1`.** In that theory the field
*sets* the strength of gravity, so `G_eff/G_N` is neither 1 nor
constant. Fitting `fsigma8` with `mu = 1` gives General Relativity's
growth attached to a modified background: finite chi-squared,
plausible posterior, no way to tell. `growth="quasi_static"` now
covers it, with the sub-horizon result of Boisseau,
Esposito-Farese, Polarski & Starobinsky (2000),
`mu = (2F + 4 F_phi^2) / (F (2F + 3 F_phi^2))`, evaluated on the
field's own solution; `F = 1` with a constant coupling gives exactly
1, which is the normalization it has to satisfy. And `Fitter` now
*warns* when a non-minimally coupled model meets growth data with `mu`
still 1 -- the one case where the answer is silently wrong.

Checked along the way and found correct: `compute_rd=True` works for
every kind of theory model, since it uses the early-universe densities
rather than the late-time `E(z)` and so never reaches past `z_init`.

### The documentation had fallen four releases behind

An audit rather than a feature, and it found things a reader would
have been misled by. The Roadmap listed **five delivered features as
still missing** -- eBOSS DR16 ELG and Lyman-alpha, Planck lensing, ACT
DR6, Planck low-l, holographic dark energy -- and was rewritten to
what is actually left. The Features list did not mention
`CosmoFit.theory` at all, the largest addition of four releases,
invisible where a reader starts. `REFERENCES.md` had nothing for the
theory module although the code cites three papers; added, with every
arXiv identifier verified against arXiv's API rather than written from
recall, and saying which is validation the module is checked against
and which is a perturbation result it imports.

## v0.38.1

Housekeeping over two capabilities the notes recorded as "should work,
but untested" -- and one of them turned out not to work at all. 484
tests.

### Scalar-tensor gravity, which four places already said worked

The minisuperspace docstring, `reduce_order`'s error message,
`fields.py` and the README all stated that a non-minimally coupled
`F(phi) R` reduces correctly. The reduction does handle it. It could
not be **written**: the namespace the `gravity` expression is parsed
in carried the cosmological parameters and the geometry scalar and no
field names, so `Action("(1 + xi*phi**2)*R", fields={"phi": ...})`
failed with `Unknown name(s): ['phi']`. An entire model class --
Brans-Dicke, induced gravity, anything where the field sets the
*strength* of gravity rather than sitting on top of it -- was
unreachable while being advertised.

Two things had to follow. **The field has to become a function of time
in the gravitational sector too**: arriving as a plain symbol,
`F(phi)` differentiates to zero, the `3 H dF/dt` term vanishes, and
what is left is a rescaled General Relativity with a Friedmann
constraint that still looks entirely reasonable. That is the same
failure the field Lagrangian had two releases earlier, which is why
the test does not only check the ratio to the textbook form -- it
asserts the constraint genuinely depends on the field's *velocity*,
which a coupling that had quietly disappeared could not produce. And
**a field cannot be named `R`, `T` or `Q`**: it would be the same
symbol as the geometry scalar in every expression, and the scalar
would silently win.

Validated three ways: the derived constraint equals
`3 F H^2 + 3 H dF/dt = rho` symbolically, ratio exactly 1; `xi = 0`
returns LCDM to 1e-9 with the closure recovering `3 Omega_de0`; and
`xi` of either sign moves `E(z)` by more than 1e-4 with constraint
drift below 1e-10.

### Multi-field actions

The other untested claim, and this one already worked. Two exponential
quintessence fields reduce to two equations of motion, integrate
together, and leave the closure condition one parameter to solve for
-- `E(0) = 1` to 1e-9, drift 2.6e-13. Now tested rather than assumed.

## v0.38.0

Removes the theory module's most conspicuous refusal -- the one you
were most likely to hit first, since `"R + alpha*R**2"` is the obvious
thing to type. A general `f(R)` builds, with nothing to ask for:
`Action` detects that the action is nonlinear in `R` and routes there
itself. 478 tests.

### Why it needed a different reduction

The ordinary reduction removes the `addot` in the Einstein-Hilbert
term by integrating by parts, which is legitimate only while the
Lagrangian is *linear* in it. For a general `f(R)` it is not -- the
term that would be dropped is not a total derivative, and dropping it
returns a different theory with nothing to show for it.

The way round is to stop treating `R` as shorthand for a combination
of `a` and its derivatives and make it an **independent variable**,
held to that combination by a Lagrange multiplier. Varying the
multiplier returns `R = R_geom`; varying `R` gives
`lambda = N a^3 f'(R)`; substituting back leaves
`L = (1/2) N a^3 [f(R) - f'(R) R + f'(R) R_geom]`, which is linear in
`addot` again. So `reduce_order` applies unchanged, at the cost of one
extra dynamical variable -- and that variable *is* the theory's fourth
order. It surfaces as a parameter, **`R_0`**, the Ricci scalar today:
an initial condition General Relativity does not have.

What comes out is smaller than "fourth-order" suggests. Varying `R`
returns `R = 6(2H^2 + H H')`, an explicit `H'`; the constraint is
linear in `R'`, giving an explicit `R'`. Two first-order equations in
`(H, R)`.

### It integrates backwards, and `theory.fields` does not

Not an inconsistency, and measured rather than asserted. A scalar
field integrated backwards runs away -- the Hubble friction that damps
it forwards becomes anti-friction. The `f(R)` scalaron does not: a
1e-8 relative kick to `R_0` moves `E(z)` by *less* than that out to
z = 1100, amplification below 1. So there is no shooting here and no
early initial condition, and `closure=` is refused rather than
accepted and quietly ignored -- there is nothing left for it to fix.

### The accuracy measure had to change too

In `theory.fields` the Friedmann constraint is a *first integral* that
the integration never imposes after the initial conditions, so its
drift is an independent measure of the error. Here the constraint is
one of the two equations that **define** the right-hand side: its
residual is zero by construction and measures nothing at all. A drift
check copied across would have looked reassuring and said nothing.
What is left over is the *third* equation of motion, from varying `a`,
which follows from the other two by the Bianchi identity -- it holds
on an exact solution and drifts on an approximate one. It comes out at
1e-15.

Validated three ways: the derived constraint equals the textbook
`3 f_R H^2 = (f_R R - f)/2 - 3 H d(f_R)/dt + rho` symbolically;
`alpha -> 0` approaches LCDM *smoothly* rather than merely resembling
it at one point (6.4e-01, 1.8e-01, 4.2e-03 for alpha = 1e-1, 1e-3,
1e-5 over z <= 5); and the routing boundary holds -- `R - 2*Lam` stays
on the ordinary path, any nonlinearity in `R` does not, and
`f(T)`/`f(Q)` are never diverted.

`dR/dN` carries `1/f''(R)`, which vanishes in the GR limit, so a model
very close to GR is **stiff -- slow rather than wrong**: alpha = 1e-3
integrates in about 3 ms, alpha = 1e-9 in about 650 ms. Measured, and
said in the error message when the integrator gives up.
`growth="quasi_static"` is refused for `f(R)`, whose `mu` is
scale-dependent -- a Compton wavelength enters -- so the scale-free
`1/f'` of the teleparallel sectors would be wrong rather than
approximate. `FRHuSawicki` already carries the standard form.

## v0.37.3

One notebook, closing the gap the previous release ended by naming.
Seventeen notebooks; 463 tests.

`planck_lite` -- the CMB computed from scratch rather than compressed
to three numbers -- had exactly one example, and it was a research
analysis. Fine if that is your question; no use at all for finding out
what the dataset *is* and whether you want it.
`01-getting-started/cmb_from_scratch.ipynb` is the ten-minute version,
and makes four points, each with a measurement.

**What it costs**: 0.2 ms per evaluation for the three compressed
numbers against **766 ms** for the 613 computed bandpowers -- three
orders of magnitude, measured live and on *new* parameter points so
the CAMB cache does not flatter the result. Which is why the notebook
uses `best_fit` and a Fisher matrix and says plainly that an MCMC here
takes days.

**What it refuses**: CAMB solves the perturbation equations of General
Relativity, so a modified-gravity model is turned away rather than
quietly handed GR perturbations.

**Why it cannot pin `tau` on its own**: `plik_lite` starts at l = 30
and reionization's signature lives below it, so the high-l spectrum
measures `A_s e^{-2tau}` and not its two factors. Doubling `tau` alone
costs **+4562** in chi-squared; doubling it with the amplitude
following costs **-5** -- nothing, and if anything preferred. That
contrast is the entire argument for adding `"tau"` or
`"planck_lowe"`, and it is two lines of output.

**That it works**: six parameters from TT/TE/EE and a Gaussian `tau`
prior alone land inside 0.7 sigma of Planck 2018 on every one, and a
Fisher matrix reproduces Planck's published uncertainty budget in 60
seconds -- ratios 0.93 to 1.01 across all five -- where a chain would
have taken days.

## v0.37.2

About the **examples**. The library had gained models from an action,
scalar fields, evidence, profile likelihoods, tension statistics and
three holographic models, and `examples/` showed none of it: nine
notebooks in a flat directory, covering the analyses the library had
grown out of rather than what it can do. Sixteen notebooks now, in
five sections with an index. 463 tests.

### Seven new notebooks

Every one executed end to end against real data, with short chains
(~3000 steps, except where the model itself is expensive and the
notebook says why).

* **`custom_models`** -- the three routes to your own `E(z)`: an
  expression string, a function, a `Cosmology` subclass. Plus
  `mu(a,k)` for modified gravity, and what a singular Fisher matrix
  tells you.
* **`models_from_an_action`** -- rederives LCDM to 2e-16 and
  `FQExponential`'s transcendental constraint to 1e-15, fits a
  power-law `f(T)` the library does not provide, and demonstrates all
  three refusals with their actual messages.
* **`scalar_field_models`** -- both Copeland-Liddle-Wands attractors
  reproduced to five decimals, the wall where a steep potential cannot
  reach the requested `Omega_m`, and why the field's initial
  conditions go early rather than today.
* **`holographic_family`** -- HDE, ADE and RDE against
  arXiv:2607.09732, including that paper's split between probes, which
  falls out without being used as an input: supernovae pull `c` to
  1.183, BAO to 0.930, combined 0.999.
* **`evidence_and_model_selection`** -- and the three tools disagreed,
  which was not staged. wCDM against LCDM on CC+DESI+Pantheon+:
  `Delta chi2 = 3.46` for one parameter (p = 0.063), **AIC favours
  wCDM, BIC favours LCDM, and `ln B = -0.83` prefers LCDM** -- the
  opposite direction from `Delta chi2`. Nothing is wrong with any of
  them; they answer different questions.
* **`profile_likelihood_and_fisher`** -- Fisher against MCMC (ratios
  0.87 to 1.07 here), profile against marginal, and why a parameter on
  a boundary breaks the usual `Delta chi2` reading.
* **`tension_statistics`** -- four definitions of "they disagree at 4
  sigma": Gaussian, sample-based, N-dimensional, and suspiciousness,
  which divides the prior dependence out.

### `dataset_zoo` covers all 21 datasets

It covered eight, and said "all eight" -- written when that was true
and left behind by thirteen additions since. Three things the old one
could not say, because everything it knew about was Gaussian: that
**three datasets ship a likelihood surface** rather than a mean and a
covariance; that **a tabulated likelihood's chi-squared has an
arbitrary zero point** (`planck_lowe` reads 396 for 28 points and is
not a bad fit -- only differences in it mean anything); and that there
are **sixteen conflicting pairs**, against the two the old notebook
named by hand, read from `CONFLICTING_DATASETS` itself so the notebook
cannot drift from the code.

### Three bugs the examples turned up

All of the kind that only surface when someone follows the
documentation rather than the tests.

**A direct `Cosmology` subclass could not be constructed.** The
distance integrator interpolates `1/E(z)` with a Hermite spline built
from `dEdz` -- that is what makes it exact to fourth order instead of
second -- so it needs the derivative from `__init__` onwards.
`define_model` installs a central-difference fallback; subclassing
deliberately does not, so a model can supply the exact derivative.
That choice was made when `dEdz` only fed the deceleration plot, and
the switch to Hermite splines silently promoted it to a hard
requirement. Nothing said so: the abstract methods raised a bare
`NotImplementedError` from inside `__init__`, nowhere near the missing
method -- **and the subclassing example in the README did not run.**
Fixed in both directions.

**An invalid value in an expression model arrived as
`KeyError: '__import__'`.** `model_from_expression` evaluates with
builtins removed, and an optimizer restart landing where the
expression goes invalid is fine -- `E(z)` comes back non-finite,
chi-squared does too, the point is rejected. What it could not survive
was numpy *reporting* it: emitting a `RuntimeWarning` runs
`warnings.warn`, which needs the builtins that were removed. Fixed by
evaluating inside `np.errstate(all="ignore")`; adding `__import__`
back would have opened the door the empty builtins dict exists to
close.

**The eBOSS Lyman-alpha likelihood could not be evaluated at all under
SciPy >= 1.18.** That release changed `RectBivariateSpline.ev` to
return a length-1 array where a scalar query previously gave a 0-d
one, so `float(log_likelihood())` raised `TypeError`. The 3-D branch
already restored the caller's shape; the 2-D one now does too.

## v0.37.0

Finishes the piece the previous release deliberately left out.
`CosmoFit.theory` could already *reduce* an action with scalar fields,
but `build()` refused, because the coupled system has to be integrated
rather than solved redshift by redshift. It no longer refuses. Each
field adds two parameters: its value and `dphi/dN` at `z_init`. 458
tests.

### Where the equations come from

Nothing further was typed in. The Friedmann constraint is a **first
integral** of the equations of motion, so rather than separately
varying `a` to get the acceleration equation, this differentiates the
constraint along the solution and solves that together with the field
equations. The two routes are equivalent by the Bianchi identity, and
this one reuses expressions already in hand -- and leaves the
constraint as an *independent* check, imposed only in the initial
conditions and never again, so its drift measures the integration
error rather than restating the tolerance. It stays at 1.7e-13.

### A design decision, got wrong first

The field's state is set at `z_init` (default 3000), not today. The
first version set it at `a = 1`, which is far more convenient: `H = 1`
holds there by the definition of `H0`, so `E(0) = 1` is algebraic and
no shooting is needed at all.

It is also wrong. Integrating **backwards** from a field at rest
today, the Hubble friction that damps the field forwards in time
becomes anti-friction, the generic past solution is kinetic-dominated,
and `rho_phi` grows as `a^-6`. For exponential quintessence normalized
to today's dark-energy density that gives **`E(2.5) = 9.3`, where LCDM
gives 3.7.** Nothing about the calculation fails. It is the correct
past of those initial conditions, and those initial conditions are not
a universe. This is the same trap `ADE` was rewritten out of one
release earlier -- worth recording as a pattern rather than an
incident.

So the state is given where a quintessence model actually gives it,
early and typically frozen, the system is integrated forwards, and
`E(0) = 1` becomes a **shooting** condition on `closure=`. Forwards is
also the numerically stable direction.

### Validation: Copeland, Liddle & Wands (1998)

Their late-time attractors for an exponential potential are exact
functions of the slope alone, so there is nothing to tune. On a matter
background there are two, selected by `lambda^2` against 3, and
**both are reproduced to five decimals, from the action**: the
field-dominated attractor `w = -1 + lambda^2/3` at lambda = 0.5, 1.0,
1.5, and the **scaling** attractor at 1.9 and 2.0, where the field
tracks the matter it scales against at a fixed fraction
`3/lambda^2`. That fraction is the sharp test -- a pure prediction of
the slope with no freedom left anywhere in the model.

It is also why a steep potential cannot accelerate: ask such a model
for more dark energy than its attractor allows and `H(0)` saturates
below 1 with no potential scale reaching it. That now arrives as a
statement about the model rather than as a stalled integrator. A
constant potential pins the other end -- a cosmological constant
written as a field, reproducing `LCDM` to 1e-10.

### Two things measured rather than assumed

**Comparing integrators at equal tolerance is the wrong comparison.**
Radau looked clearly best that way. At equal *accuracy*, which is the
comparison that decides anything, DOP853 reaches a tighter drift in
**3.4 ms against 79**.

**`w` and `Omega_de` must be read off the field's Lagrangian**, not
off the background. The generic route -- subtract the fluids from
`E(z)^2` -- cancels catastrophically once matter dominates: at
z = 2000 it is 2.4e9 minus 2.4e9 to get 0.7, nine digits gone. It was
returning `w = -5.8` where the answer is -1. With `p = L` and
`rho = 2 X L_X - L` there is nothing to cancel. The fieldless models
keep the background route, since a modified gravitational sector has
no Lagrangian pressure to read, and its docstring now says where it
stops being trustworthy instead of leaving that to be discovered.

About 37 ms per parameter set, down from 58 by warm-starting the
shooting from its previous solution and carrying the secant slope with
it. Grid density turned out to be free -- DOP853's dense output is
cheap and the cost is all in the steps -- so it is set for accuracy,
5e-12 relative in `E(z)`.

## v0.36.0

Where the last release was entirely about speed, this one is about
**what the library can say**: a way to build a model from its action
rather than from an `E(z)`, the inference tools needed to judge
whether a model's improvement is real, and two new dark-energy models.
20 models (was 18), 439 tests (was 354).

### Models from an action

Every model in `cosmology.models` encodes the *result* of somebody's
derivation: an `E(z)` transcribed from a paper. `define_model` lowered
the barrier to adding one but not the work -- it still wants `E(z)`.
`CosmoFit.theory` takes the input instead. What comes back is an
ordinary `Cosmology` subclass, so every dataset, likelihood, sampler
and plot works on it unchanged; `Action.constraint()` returns the
derived Friedmann equation symbolically, if the derivation was the
point rather than the fit.

**The lapse is the whole trick.** FLRW is written with an explicit
`N(t)`, the action is reduced to a point-like Lagrangian in `a`,
`adot` and `N`, and varying the lapse gives the Friedmann
*constraint* -- `N` is a non-dynamical gauge degree of freedom, which
is precisely why its variation produces a constraint rather than an
evolution equation. Setting `N = 1` afterwards recovers the familiar
form; writing `N = 1` from the start loses the equation altogether.

It rederives what the library already had by hand: `R - 2*Lam` gives
`LCDM` including curvature to **1e-16** in `E(z)` and `dE/dz`;
`Q*exp(lam*Q0/Q)` gives `FQExponential` with an identical constraint
and agreement to **1e-13**; and `T + A0*(-T)**b` at `b = 0` gives
`LCDM` again through the torsion sector. `Lam = 3 Omega_de0` is
written nowhere -- it falls out of the closure condition. The `f(Q)`
row is the demanding one: its constraint is transcendental and the
hand-written model inverts a Lambert `W`, so the derivation has to
produce the transcendental form from the action alone *and* the solver
has to land on the branch `W_0` picks rather than on any root of it.

It refuses rather than approximates. A general **`f(R)` is refused**
(fourth-order; the integration by parts that removes `addot` would be
discarding something that is not a total derivative). An action that
does **not satisfy `E(0) = 1` is refused** -- it would predict every
distance wrong by a constant factor while looking entirely healthy.
And **`growth="quasi_static"` is opt-in**, because `mu = 1/f'` is a
statement about perturbations, which a background action does not by
itself determine. The three geometry scalars carry different signs
across the literature, and choosing wrongly inverts every modification
built on one with nothing downstream to complain, so they are fixed by
requiring an undeformed `f` to reproduce General Relativity exactly --
asserted in all three sectors rather than trusted.

Two bugs the tests caught, both a correct-looking Friedmann equation
hiding wrong physics: the scalar field arrived as a plain sympy
`Symbol` rather than `phi(t)`, so `V(phi)` differentiated to zero and
the Klein-Gordon equation was wrong while the constraint looked
perfectly reasonable; and a root-finder turned loose on the whole
redshift range can switch branches of a transcendental constraint and
return a discontinuous `E(z)` **without failing**, hence solving by
continuation out from z = 0 where the closure condition guarantees the
root.

`sympy` is a new optional extra; CI asserts the minimal install still
works without it.

### Bayesian evidence, profile likelihoods and Fisher matrices

Three things the notebooks kept building by hand or disclaiming:
`stats.nested` (dynesty), `Fitter.profile()` and `Fitter.fisher()`.
And the first thing they did was **change a verdict**. LsCDM's
`Delta chi2 = 5.40` over LCDM had looked like real evidence; its Bayes
factor is `ln B = +0.494 +/- 0.319` -- **inconclusive**. The reason is
visible in the posterior: the transition redshift's 95% interval runs
to [2.5, 96.6], so most of its prior volume does nothing and the
Occam factor eats the improvement. That prior sensitivity is now
*measured* rather than asserted -- narrowing the range gives
`ln B = +1.158`. `stats.tension` adds Gaussian 1-D/N-D, sample-based
and suspiciousness statistics.

Four bugs these surfaced, three of them found by an answer that was
*impossible* rather than merely surprising:

* **`best_fit()` settled in local minima** -- found by LsCDM fitting
  *worse* than the LCDM it contains, by `Delta chi2 = -0.63`.
  `best_fit(restarts=n)` added.
* **`best_fit()` left the cosmology at the last evaluation**, not at
  the best fit -- which made a whole chi-squared breakdown table read
  +0.00 everywhere.
* **A NaN parameter vector crashed from inside `refresh()`.** L-BFGS-B
  started where chi-squared was inf, computed `inf - inf` for the
  gradient, and evaluated at `[nan, nan, nan]`; `_apply()` sat outside
  `chi2`'s try block.
* **`likelihood_ratio_test` reported an impossible result as "no
  evidence"** -- `chi2.sf(negative) = 1.0` and `norm.isf(1.0) = -inf`,
  indistinguishable at a glance from a genuine null. It now warns,
  with a tolerance so nested-sampling convergence noise does not trip
  it.

### A free neutrino mass now warns when nothing can see it

The compressed Planck distance priors are **blind to `m_nu` by
construction**: their `z_star` is a fitting formula calibrated at the
fiducial 0.06 eV and returns the same `z_star`, `Omega_r` and
`r_s(z*)` anywhere from 0 to 0.8 eV. A free `m_nu` there runs to 0.82
eV meaninglessly, and a posterior for it is not a neutrino-mass
measurement. The fit now says so and names what to add instead. Run
properly with the full CMB, the 95% profile bound is 0.039 eV against
a published Bayesian 0.064 eV, and the three candidate reasons for the
gap are flagged as unresolved rather than papered over.

### Agegraphic and Ricci dark energy

Two more holographic models, validated against the same paper as
`HDE`. RDE's `gamma = 0.538` and `Omega_m0 = 0.2173` sit inside both
published intervals; ADE's `n` is ~2% high, and the honest reading is
that the model is right and the fit differs -- `n = 2.80` predicts
`Omega_m = 0.280`, exactly the pair that paper reports.

**ADE has one fewer free parameter than LCDM.** Its early-time
condition fixes the background from `n` alone, so `Omega_m` is not fit
but **derived**. New generic `DERIVED_PARAMS` declaration: a model
lists what it computes, and the fitter warns if one is set free, since
sampling it would just be reintroducing the prior.

Both were first written wrong in ways the tests caught. ADE was
integrated *backwards* from today with `Omega_m` free, which is
silently a different model. And RDE's `Omega_de(z)` returned the
second term of `E^2`, but the Ricci density carries a matter-like
piece, so reading the two terms as "matter" and "dark energy"
misplaces about a third of the matter density. A third, from the
previous release: `HDE.Omega_de(z)` returned the density *parameter*
`rho_DE/rho_crit(z)` where the rest of the library means
`rho_DE/rho_crit(0)` -- the two agree at z = 0, which is why it
passed, and are off by 0.59x at z = 0.5.

## v0.32.0

Five commits, all of them performance. No new datasets, no new models,
no API changes -- the same 21 datasets, 18 models and 354 tests,
running between **1.8x and 29x faster** depending on the fit. Two of
the changes also made the code **more accurate**, which is the part
worth reading.

### The profile was not where I expected

A growth fit with **68 data points** cost 27 times what one with
**1869** did. That was the whole finding, and everything else followed
from measuring rather than guessing.

**The growth ODE.** `solve_ivp` at `rtol=1e-8` called back into Python
for every stage of every step: 19,040 scalar right-hand-side
evaluations per solve, each evaluating `E(z)` three times. But the
equation is linear, smooth and non-stiff, so adaptivity was only ever
rediscovering a step size that never needed to change. It is now a
fixed 300-step RK4 with the coefficients evaluated for the whole grid
in four array calls.

**The stepping itself, twice.** RK4 is sequential, so there was no
array operation left to hand NumPy -- 493 microseconds of pure loop.
Two ways round it, both implemented. *Without numba*, the equation
being **linear** means one RK4 step is a fixed 2x2 matrix and
composing them is a prefix product, which pairwise doubling does in
log2(n) rounds of batched work: 493 to 215 microseconds. *With numba*,
the same loop compiled: **5.2 microseconds**. The `speed` extra was
already declared and nothing used it; it does now. The two agree to
8.9e-16, which a test asserts by running both, and CI covers both
paths.

**`x ** 3` goes through `pow`.** NumPy special-cases `x ** 2` and
`x ** 0.5`; it does not special-case cubes. On a 505-element array
that is 5.02 microseconds against 0.79 for `x * x * x` -- and
`Omega_m * (1 + z) ** 3` is the most-evaluated expression in the
library. Applied across seventeen files; `E(z)` dropped from 9.37 to
5.23 microseconds, and the rewrite differs from `pow` by one unit in
the last place.

**Building the Hermite splines directly.**
`CubicHermiteSpline.__init__` validates, sorts and normalizes axes
before assembling coefficients that are four lines of algebra: 43
microseconds against 13. `cosmology.refresh()` -- paid by every fit on
every step -- went from 99 to 48.

### Two places this got more accurate, not less

**The distance integral.** `chi(z)` was a plain trapezoid fitted with
a shape-preserving Pchip, which spent most of its build time deriving
slopes that are known exactly, since `chi'(z) = 1/E(z)`, and got them
slightly wrong. It is now a corrected (Euler-Maclaurin) trapezoid and
a Hermite spline built from that same derivative. Against `quad`, the
maximum relative error on the default grid falls from **4.5e-06 to
1.3e-10**.

**The recombination and sound-horizon integrals.**
`scipy.integrate.simpson` accepts an arbitrary `x` and pays for that
generality whether or not the grid is uniform -- 36 microseconds of a
75 microsecond integral. Every grid here comes from `linspace` or
`logspace`, so uniformity is known at the call site. `r_s(z*)` goes
from 2.5e-07 to **2.8e-09** and `r_d` from 3.2e-09 to **3.4e-11**,
both roughly twice as fast. A uniform-grid rule on the substituted
integrand is simply a better rule than a general one on the original.

### What was measured and rejected

**jax** -- its win comes from JIT-ing and fusing a whole computation,
which would mean rewriting every model's `E(z)` in jax primitives.
That is a rewrite, not an optimization. **A Cholesky factorization of
the supernova precision matrix** -- exact and worth 1.23x, but it
costs 106 ms at dataset load against 131 ms for the entire load, so it
makes short runs and the test suite slower to make long chains faster.
**float32 for that matrix** -- 1.3x, for 1.8e-05 absolute in
chi-squared; not for sale in a library whose recent conclusions turned
on a `Delta chi2` of 0.30. **A symmetric `dsymv`** -- 15x *slower*,
because SciPy copies the C-contiguous matrix to Fortran order first.

### A wrong turn, recorded

cProfile attributed ~150 microseconds per evaluation to `simpson`. I
wrote a direct replacement, found it worth 1.4, concluded the profiler
had been inflated by its own per-call overhead, and reverted. That
conclusion was wrong, and the check was the reason: I had timed
`simpson(y, dx=h)`, a different and three times faster code path than
the `simpson(y, x=grid)` the callers actually used. The profiler had
been right, and finding that later produced the last commit in this
release -- the one that also improved the accuracy.

## v0.31.0

Mostly about **BAO that is not a Gaussian**, one new model, and four
bugs -- three of which were found by an answer that was impossible
rather than merely surprising. 21 datasets, 18 models, 317 tests.

### BAO released as a likelihood surface

Every BAO dataset here used to be a mean and a covariance. Three of
eBOSS DR16's are not, and in each case the collaboration released a
grid because a Gaussian would misrepresent the measurement.

**`eboss_elg`** -- `D_V/r_d` at z = 0.845, a 399-point curve. The BAO
feature is a **1.4 sigma detection**, so the likelihood is asymmetric
and still *rising* at the low edge of the released table. About a
tenth of its probability sits below `D_V/r_d = 16.5`, where a Gaussian
at 18.33 +/- 0.6 puts a thousandth -- two orders of magnitude, in
exactly the tail that decides whether a low expansion rate at
z ~ 0.85 is excluded.

**`eboss_lya`** -- `(D_M/r_d, D_H/r_d)` at z = 2.334, a 50x50 surface;
the highest-redshift BAO in the library outside the CMB, and
independent of DESI's.

**`eboss_elg_fs`** -- the ELG sample again, full shape: a
**100x100x100** grid. Three dimensions is the point rather than an
inconvenience, since `fsigma8` is degenerate with the Alcock-Paczynski
distortion and a grid carries that exactly where two error bars could
not.

All validated against the papers with the published numbers used
nowhere as inputs. The ELG pair is more than a central-value check: it
identifies *which* interval the paper quotes -- that table's standard
deviation is 1.05 and its 68% credible interval is +0.52/-1.21, and
only the `Delta chi2 = 1` crossings give the published asymmetric
pair. The Lyman-alpha check does double duty: eBOSS quote the combined
constraint from a joint fit but ship the auto- and cross-correlation
surfaces separately, so this multiplies them, which assumes
independence -- recovering the published *errors* from the product is
what makes that defensible, since a neglected correlation would have
come out as errors too tight.

Two things this cost, both now tests. Interpolating the released
probability directly does not work -- thirty orders of magnitude, so a
cubic is dominated by the peak and undershoots to negative values that
have no logarithm. And for the 3-D grid, splining the *log* also
fails: it ships floored 200 below its peak and a cubic rings at the
step where that plateau starts, reaching a log of **+146** against a
node maximum of 0. That path is linear, and a test asserts the
no-overshoot property directly.

### `sdss_fsbao`, and a gap it uncovered

The SDSS BAO **+ full-shape** consensus: `D_M/r_d`, `D_H/r_d` *and*
`fsigma8` at four redshifts, with the covariance between them.
Reproduces Alam et al. (2021) Table 3 exactly, and the errors come
from a covariance file that is a separate product, so that also checks
the two were paired correctly. The real content is the correlation:
within a bin, `D_M/r_d` and `fsigma8` are correlated at +0.388,
+0.389, +0.185, +0.636.

Which is the gap. **`sdss_bao` + `fsigma8` had been reachable without
a warning all along** -- the growth compilation includes BOSS z = 0.38
and the eBOSS quasars, so the pair covers the same galaxies twice
while treating geometry and growth as independent. Now registered as
conflicting, with the joint product named as the alternative.

DESI full-shape is *not* here and cannot be on this library's terms:
DESI publish MCMC chains and the full modelling-pipeline inputs but no
compressed Gaussian summary, so using it properly means implementing
an EFT model with its nuisance parameters. That is well outside a
background-plus-linear-growth library and could not be validated
cheaply.

### Holographic dark energy

The holographic principle bounds the energy in a region by its
boundary area, giving `rho_DE = 3c^2 M_p^2 / L^2`; Li (2004) showed
`L` has to be the **future event horizon** for the universe to
accelerate at all. Architecturally it is the first background model
here **without a closed-form `E(z)`** -- `Omega_DE` obeys an ODE,
solved and splined on every `refresh()`.

Validated against the model's *definition* rather than its equation of
motion, which is the point: for an ODE model there is nothing to check
by inspection, and a wrong equation gives a smooth, plausible,
entirely wrong history. The test computes the future event horizon by
quadrature from the solved `E(z)`, with the ODE appearing nowhere in
it, and checks `H(a) L(a) = c / sqrt(Omega_DE(a))`. It holds to 2e-3
for `c` from 0.6 to 1.3, and a second test deliberately corrupts the
ODE and asserts the check notices, so it cannot quietly become
vacuous. That check is also why the solution runs *forward* to
a = 1e5: nothing asks for z < 0, but a model defined by the future
should be answerable about it. Flat only, and it says so -- curvature
changes the causal structure the holographic bound is applied to.

### Four bugs

`best_fit()` settling in local minima; a NaN parameter vector crashing
from inside `refresh()`; `likelihood_ratio_test` reporting an
impossible result as "no evidence"; and that warning then firing on
convergence noise -- a 32-fit scan returned five negatives between
-8e-06 and -5e-04, every one at the nested limit where the two
optimizers reach the same minimum by different routes, and warning
about those trains the reader to ignore the warning.

### Two notebooks

**The LsCDM result in `lscdm_mcmc.ipynb` is a DESI result.** Section 8
had found the transition-redshift constraint as a *cliff* --
chi-squared falling by 28 between adjacent grid points, straddling
DESI's Lyman-alpha bin at z = 2.33 -- and could not say whether that
was a property of the universe or of one measurement. With eBOSS's
independent Lyman-alpha at essentially the same redshift, it can: the
`Delta chi2 = 5.40` preference becomes **0.30**, and the 28.04-unit
step becomes a 1.6-unit ripple. The mechanism turned out not to be
what the cliff suggested -- broken down by dataset, DESI's 5.40 is
2.72 in Planck and 2.12 in DESI's own BAO, and with eBOSS the Planck
term is already 0.10: no tension, nothing to relieve.

**`dark_energy_evidence_audit.ipynb`** -- the same question asked
twenty ways. LsCDM's improvement is a compound of all three choices:
0.00 with no BAO, 0.30 with eBOSS, and -- the sharpest -- **5.40 drops
to 0.81 when `r_d` is fitted rather than computed**. The evidence lives
in a CMB-calibrated BAO ruler disagreeing with the CMB's own distance
priors; remove the calibration and there is nothing to relieve. CPL's
is smaller and does not depend on any single choice, and all twenty
cells land on `w0 > -1` and `wa < 0`.

## v0.28.0

`v0.25.0` computed the CMB; this **finishes** it, adds a second
telescope, and closes the loophole that let the whole thing be tested
against nothing. 230 tests (from 188), 17 datasets.

### The Planck 2018 likelihood, all four parts

`v0.25.0` shipped one component of four. The CMB is now complete --
**652 data points**, each part validated against a published number
rather than against itself: Commander low-l TT (data shipped, never
read), `plik_lite` high-l TT/TE/EE (already there), SimAll low-l EE
(stood in for by a Gaussian `tau` prior) and lensing (missing
entirely).

**Low-l EE is the real thing, not the `tau = 0.0544 +/- 0.0073`
shorthand.** Below l = 30 there are only 2l+1 modes on the sky, the
distribution is strongly skewed, and that regime carries essentially
all of the CMB's information about `tau` -- so Planck ships a
probability table and the likelihood is a lookup, with no mean and no
covariance. It is the library's first non-Gaussian likelihood, and
`BaseLikelihood` no longer assumes every dataset has a covariance to
invert. Validated **by reconstruction**: profiling `tau` against that
table plus the high-l bandpowers returns **0.0541 +/- 0.0072** against
Planck's published 0.0544 +/- 0.0073, with that number used nowhere as
an input, and the fitted amplitude tracks `tau` with slope 2.00, which
is the `A_s e^{-2tau}` degeneracy exactly.

**Lensing** is the CMB's own growth measurement -- every other CMB
dataset here constrains recombination and reaches the present only
through a distance. It is also the part that is easy to drop: the
reconstruction is *normalized* against an assumed cosmology, so the
linear correction propagating that dependence vanishes at the fiducial
by construction. A chi-squared check at Planck's best fit therefore
passes whether or not it was implemented, and the tests check it away
from there.

### ACT DR6 lensing

A second, independent reconstruction -- different telescope, different
sky, different pipeline, and tighter than Planck's: **2.3% on the
lensing amplitude**. Fitting a single scaling of the theory at
Planck's best-fit LCDM returns `A_lens = 1.017 +/- 0.026` against
ACT's published 1.013 +/- 0.023, with Planck's own lensing at the same
cosmology giving 0.998 +/- 0.025.

That test earns its place because the failure mode is a **smooth
rescaling**. ACT's products are built on the lensing convergence,
Planck's on the potential, and the two differ by 2 pi / 4. Get it
wrong and nothing crashes: the theory comes out uniformly off, a fit
absorbs it into the amplitude, and the posterior looks ordinary and
sits somewhere else -- `A_lens` would read 0.65 or 1.6 instead of 1.

### sigma8 derived instead of fitted

`sigma8` was defined **twice** -- the free parameter the growth
machinery normalizes with, and CAMB's derived value from the primordial
amplitude -- and nothing made them agree.
`Fitter(derive_sigma8=True)` takes the amplitude from the Boltzmann
code and drops it from the sampled parameters. This is what makes the
CMB *predict* growth rather than accommodate it: with `sigma8` free,
an `S8` measurement is reproduced to chi-squared ~ 0 and nothing has
been tested, because a free parameter slid onto it.

### `best_fit()` was returning its starting point

The most serious bug in this release, and it was silent. L-BFGS-B
estimates gradients by finite differences with a step of ~1.5e-8. For
a likelihood whose own numerical noise exceeds the chi-squared change
such a step produces -- which is every CAMB-based likelihood -- the
estimated gradient is zero and **the optimizer returns the starting
point while reporting `success=True`**. The caller gets their initial
guess back labelled "best fit", every downstream number is computed
from it, and nothing anywhere says so. On the full Planck CMB it cost
**366 in chi-squared**.

`best_fit()` now detects the stall -- movement below 1e-6 of each
parameter's own prior width, which is the only scale-free way to ask,
given that `H0` and `Omega_b` differ by three orders of magnitude --
and retries with Nelder-Mead on a prior-scaled simplex, keeping the
better of the two. Passing `eps=` or `method=` suppresses the rescue:
the caller has taken control. Deliberately narrow, because a larger
step is *not* uniformly better -- on CPL with CC+DESI+Planck it finds
a minimum 2.75 worse.

### Infrastructure

**Continuous integration**, finally. Two jobs: the full suite with
every extra across Python 3.11-3.13, and a `minimal` job installing
core dependencies only -- the configuration a plain
`pip install cosmofit` produces. The second is not redundant: CAMB and
Streamlit are optional and every test needing them skips, so it is the
only thing checking that the *skip paths* skip rather than erroring.
It asserts both extras really are absent first, so a leaked transitive
dependency cannot turn it into a duplicate of the first.

**The neutrino-mass path is now pinned.** It worked when it was built
and no test touched it. It has two consumers that never talk to each
other -- in the sound horizon the neutrinos are relativistic at the
drag epoch and *raise* `r_d`; in CAMB they free-stream out of the
small-scale power and *lower* `sigma8` -- and both read the same
`Omega_m`. Getting the subtraction right in one place and wrong in the
other would not raise; it would move the two in opposite directions by
a fraction of a percent, comfortably inside the range where a fit
still looks plausible and is wrong.

### `examples/s8_tension_cmb.ipynb`

The S8 tension posed from the CMB side, using everything above: all
652 Planck points, `sigma8` derived, and **neither weak-lensing
measurement in the fit**. The fit reproduces Planck's published
parameters to within 0.16 sigma, which is the bar it had to clear
before anything else means anything, and predicts
`S8 = 0.8307 +/- 0.0104` -- **2.9 sigma from KiDS-1000, 2.7 sigma from
DES Y3**. ACT DR6's lensing never entered the fit, so its `A_lens` is
an out-of-sample check on the whole chain, Boltzmann code included.
The `sigma8`-free version of the same question returns chi-squared =
0.000000 and the tension is invisible. And the growth compilation
gives chi-squared/N = 0.79 against the same prediction -- **the
disagreement is in weak lensing, not in redshift-space distortions.**

## v0.25.0

A pass over the graphical interface, on the
principle that a tool which cannot be used wrongly is worth more than
one with more knobs.

The app had grown to fourteen datasets and seventeen models presented
as a flat checkbox list and a flat dropdown, with no indication of
what any of them was. That is a problem specific to this domain: a
wrong combination does not error, it returns a perfectly
ordinary-looking posterior with error bars that are too small, or a
"measurement" of a parameter nothing in the fit constrains.

**Everything now explains itself.** Each dataset says what it
measures, over what redshift range, how many points, what it actually
constrains, and where it comes from -- grouped by probe, since a fit
is built by taking one from each family rather than by ticking
everything. Each model says what it is, what its extra parameters
mean, and which parameter values reduce it to ΛCDM, plus capability
badges for `w(z)`, growth, and whether the full CMB spectra can be
computed for it.

**Six presets** configure a whole analysis at once. "DESI DR2 + BBN →
H₀ without the CMB" sets the datasets, selects DR2, *and* turns on the
computed sound horizon -- none of the three is useful without the
other two, and expecting someone to discover that from three separate
widgets was optimistic.

**The parameter table now shows only the parameters this fit uses.**
The shared container carries every parameter any model needs; for an
LCDM fit against CC and BAO, all but four are inert. Relevance is a
property of the fit rather than the model, so it follows the datasets
too: `rd` appears only with BAO, `sigma8` only with growth data,
`n_s`/`τ` only with the CMB spectra. The hidden ones still reach the
`Fitter` -- they are inert, not absent.

**Warnings fire before the run, not after.** One is a hard error the
app can see coming (a modified-gravity model with the full CMB
spectra). The rest are quieter and matter more, because the fit will
run, converge, and produce something that looks like a result: a
modified-gravity model with no growth data ticked, a model whose own
parameters are left fixed, a free `sigma8` nothing constrains, a
computed `r_d` with `Ω_b` held fixed.

**Three things the GUI simply could not show before:**

* **χ² per dataset**, which turns a total χ² into a statement about
  *which* dataset is in tension. On the Hubble-tension preset it reads
  DESI 19.6/13, Planck 7.2/3, local H₀ **24.5/1** -- the whole
  argument, in one table.
* **Derived quantities** (`q₀`, `z_t`, `r_d` from the densities) with
  real error bars. These have existed in `CosmoFit.stats.derived`
  since v0.13.0 and were reachable only from Python.
* **Dataset versions.** DESI DR2, the SH0ES 2024 and TDCOSMO H₀
  measurements, DES Y3's `S₈` and the Cooke BBN prior all shipped in
  v0.23.0 and were **unreachable from the GUI**, which never passed
  `dataset_kwargs`. Each dataset with more than one version now has a
  picker.

A **📖 Guide** panel lists every dataset and model in one browsable
table with the conflict rules and a short walkthrough. Both tables are
generated from the same dictionaries the widgets use, so a dataset
added to the library without a note shows up as a blank row rather
than silently.

### Two bugs, and the first GUI tests

`st.checkbox` was being given both a `value=` and a session-state
entry under the same key -- the one thing Streamlit explicitly warns
about, since the two disagree about which is authoritative. And the
preset button called `st.rerun()`, which is the obvious idiom and was
never needed here (the button sits above the widgets it writes to, so
the values land on the same pass); it was also an infinite loop
anywhere a button's click state outlives the rerun.

That second one was found by the new `tests/test_gui.py`, which is the
point of it. Streamlit's `AppTest` runs the app in-process and exposes
the rendered element tree, so **12 tests** now drive the flows a user
actually takes: a preset writes the configuration it claims to, a fit
runs end to end and renders its tables, the parameter table follows
the datasets, and each warning appears when it should. **159 tests
total**, ~18 s.

Two small public additions came out of this, both because the GUI
needed them and reaching into private state to get them would have
been worse: `dataset_reference(dataset, version)` returns a dataset's
citation without loading its files, and
`cosmology.boltzmann.supports_cmb_spectra(model)` answers "can CAMB do
this model?" as a value rather than by raising -- the backend now
routes its own check through it, so the two cannot drift apart.

## v0.24.0

Computes the BAO sound horizon `r_d` from the
physical densities instead of fitting it.

### What was wrong with fitting it

Nothing, exactly -- treating `r_d` as a free nuisance parameter is a
defensible and common choice, and it makes BAO immune to any
assumption about the early universe. It is also the reason this
library could not measure `H0` from BAO. `H0` and `r_d` enter every
BAO observable only through the product `H0 * r_d`, so with `r_d`
free they are perfectly degenerate and BAO constrains neither.

`SoundHorizon` was, until now, a nine-line stub that returned the
free parameter. It now does the integral:

    r_d = int_{z_d}^inf c_s(z)/H(z) dz,
    c_s = c / sqrt(3(1 + R_b)),  R_b = 3 omega_b/(4 omega_gamma) a

### What is actually computed, and what is not

**Computed from first principles:** photon density from `T_CMB`;
massless neutrinos; the baryon loading; the integral itself.

**Massive neutrinos, exactly.** They are relativistic at the drag
epoch (`y = m/kT ~ 0.34` for 0.06 eV) and matter-like today, and the
transition matters. The usual `[1 + (Ay)^p]^(1/p)` approximation
costs 0.05% in `r_d` at `Sum m_nu = 0.06 eV` and 0.15% at 0.6 eV --
the latter half of DESI's best BAO precision, spent on an
approximation with no reason to be there. Instead the Fermi-Dirac
energy density integral is tabulated once at import (a few ms) and
splined afterwards. The familiar `Sum m_nu / omega_nu h^2 = 93.14 eV`
shorthand is then not an input at all: the calculation **derives**
93.0378 eV, where CAMB gets 93.04.

**Fitted, and honestly labelled:** `z_drag` alone. The drag epoch is
defined by a Thomson-drag optical depth over a full recombination
history, which is a Boltzmann code's job. This takes the same route
the library already took for `z_star` in v0.19.0 -- a fit calibrated
directly against CAMB, over a 5850-point grid spanning `omega_b` in
[0.018, 0.026], `omega_cb` in [0.09, 0.20], `N_eff` in [2.0, 5.0] and
`Sum m_nu` in [0, 0.6] eV.

> Eisenstein & Hu (1998)'s `z_drag` formula is bundled for comparison
> and is **not** usable here: it gives 1020.7 where CAMB gives 1059.9,
> 3.7% low, which puts `r_d` 2.5% high -- ten times DESI DR2's best
> BAO error bar. That is not a flaw in EH98; their `z_d` was
> calibrated jointly with their own closed-form `r_s`, and the two
> halves are not separately meaningful. Splicing one of them onto a
> modern integral is exactly the convention error v0.19.0 documents
> at length, in a different place.

### Accuracy

Against CAMB's `rdrag` over that whole grid:

| | max error |
|---|---|
| the integral alone, given CAMB's `z_drag` | 2.2e-6 |
| the `z_drag` fit | 6.7e-5 |
| **end to end** | **5.0e-5** |
| end to end, within realistic priors | 1.4e-5 |

DESI DR2's single best BAO bin is a 0.24% measurement, so the worst
case is ~50 times smaller than the best data's error bar and the
typical case ~500 times. A trimmed copy of that grid ships as a test
fixture, so the comparison runs without CAMB installed.

### `r_d` depends on less than you might expect

The integral runs entirely through the radiation- and
matter-dominated eras, so **`r_d` does not depend on `H0`, on
curvature, or on the dark-energy model at all** -- only on
`omega_b`, `omega_cb`, `N_eff` and `m_nu`. Verified against CAMB,
which returns the same `rdrag` to 1e-7 across `H0` from 60 to 75.

Two consequences. It works identically for **every** model in the
library, including the modified-gravity ones a Boltzmann code will
not accept. And the cache keys on the densities alone, so an MCMC
step that moves only `w0` reuses it -- which matters, because the BAO
likelihoods ask for `r_d` once per data point.

### Using it

Off by default: switching a fitted nuisance parameter into a derived
quantity changes every BAO prediction, and that is a choice about the
analysis.

```python
fit = Fitter(
    model=LCDM,
    datasets=["desi", "omega_b"],          # BAO + the BBN prior
    dataset_kwargs={"desi": {"version": "desi2025"}},
    free_params=["H0", "Omega_m", "Omega_b"],
    initial={"H0": 68.0, "Omega_m": 0.30, "Omega_b": 0.0493},
    compute_rd=True,
)
```

`rd` must not be in `free_params` alongside it -- the likelihood
would ignore the sampled value and its "posterior" would be its
prior -- and the constructor says so rather than letting it happen.
`Omega_b` becomes the parameter BAO now needs and cannot pin down on
its own, which is what the `"omega_b"` BBN dataset is for. Running
that fit (DESI DR2 + BBN, flat ΛCDM) gives

    H0      = 68.55 +0.59 -0.60
    Omega_m = 0.2977 +0.0087 -0.0085
    Omega_b = 0.0472 +0.0008 -0.0008
    r_d     = 148.10 +1.59 -1.63 Mpc   (derived)

which lands where published DESI BAO+BBN constraints do -- an
end-to-end check of the whole chain that no single unit test
provides.

`r_d` is a *derived* quantity now, so it leaves `summary()` and joins
`z_t` and `q0` in `CosmoFit.stats.derived`:

```python
from CosmoFit.stats import derived
derived.summarize(derived.sound_horizon(fit))
```

That works whether or not the fit used it. With `compute_rd=False`
it returns what the early-universe physics *would* have predicted for
the same densities, and comparing it against the fitted `r_d` is a
real consistency test: a mismatch is the standard signature of new
physics before recombination, and one of the main ways the Hubble
tension is diagnosed.

`compute_rd` is part of the chain signature, so a chain sampled one
way cannot be resumed the other. The GUI gets a checkbox, which
un-ticks `rd` for you and points at the BBN dataset if it is missing.

### One bug this surfaced

Nothing in the growth machinery: `Fitter` now also refuses
`compute_rd=True` together with a free `rd`, which was previously
expressible and would have produced a `rd` "posterior" identical to
its prior with no indication anything was wrong.

Seventeen new tests (**147 total**) cover the neutrino
thermodynamics, the CAMB comparison, the independence claims above,
quadrature convergence, and the wiring.

## v0.23.0

The largest single addition since the library
began: six new datasets, eight new cosmological models, the first
test suite, and -- the headline -- **the CMB computed from scratch
rather than compressed**.

### Planck, uncompressed

Until now "Planck" meant three numbers: the distance priors
(R, l_A, omega_b h^2). That is fast, dependency-free and works for
every model, and v0.19.0 documents at length how badly it can go
wrong, because a compression carries the conventions of whoever
produced it and the theory prediction has to share them exactly.

`"planck_lite"` is the other end of that trade. It uses the actual
measured spectra: **613 binned TT/TE/EE bandpowers over
l = 30-2508 with their full 613x613 covariance**, compared against a
C_l spectrum computed by a Boltzmann code. No compression, no
summary statistic, no borrowed convention -- the theory prediction
is the same object the data is.

CosmoFit does not implement a Boltzmann hierarchy, and should not:
that is thousands of coupled ODEs per wavenumber through
recombination, and a pure-Python version would be far too slow for
an MCMC. A new `cosmology/boltzmann.py` translates a `Cosmology`
into **CAMB**'s parameter conventions and calls it, as an optional
dependency (`pip install "cosmofit[cmb]"`). Nothing else in the
library imports it.

Three details that decide whether this is right or subtly wrong:

- **Which models can go through it.** LCDM maps onto CAMB directly.
  Any model exposing a `w(z)` (wCDM, CPL, JBP, BA, GCG, and the new
  PEDE/GEDE/IDE) is passed through CAMB's PPF dark-energy module as
  a tabulated `w(a)` -- exact at background level, and stable across
  the `w = -1` crossing that CPL posteriors routinely visit. The
  **modified-gravity models are refused outright**. `FRHuSawicki`
  would have run happily -- its background *is* LCDM's by
  construction -- and returned LCDM's C_l with `f_R0` doing nothing,
  which is worse than an error.
- **Massive neutrinos.** CosmoFit counts them inside `Omega_m`;
  CAMB counts them separately. So `omch2` is
  `Omega_m h^2 - Omega_b h^2 - Omega_nu h^2` -- *subtracting*. Adding
  instead shifts `Omega_c h^2` by ~0.0006, about half a sigma of
  Planck's constraint on it, with no other symptom.
- **A CAMB API trap, found by testing.** `pars.DarkEnergy = obj`
  copies the object into CAMB's Fortran state, so setting the `w(a)`
  table on the Python instance *afterwards* is silently discarded --
  and CAMB then returns a perfectly valid cosmological-constant
  spectrum for a w0-wa model. Caught by asserting that CPL at
  `w0 = -1, wa = 0` reproduces LCDM *and* that CPL at
  `w0 = -0.9, wa = -0.4` does not.

**Validation.** Against the reference spectrum shipped with
`planck-lite-py`, this implementation reproduces its published
log-likelihood **exactly**: -291.33481235418026 for TTTEEE
(bit-identical) and -101.58123068722571 vs -101.58123068722583 for
TT (1e-13, matrix-inversion roundoff -- Cobaya's own plik_lite
differs from `planck-lite-py` by 2e-13 on the same number).
Independently, at Planck's best-fit LCDM it returns chi2 = 585 for
613 bandpowers.

**What it costs.** One CAMB call is ~0.7 s against ~1 ms for the
whole rest of a joint likelihood, so a chain including this is
roughly three orders of magnitude slower per step. That is not an
implementation flaw -- it is why full CMB chains run on clusters and
why compressed priors exist. Budget hours, use `n_processes`, and
save the chain. A CMB spectrum also needs `ln1e10As`, `n_s` and
`tau_reio`, which no background fit ever did; and because
`plik_lite` starts at l = 30, `tau` is degenerate with the amplitude
unless the new `"tau"` dataset is included.

> The primordial amplitude is `ln1e10As`, not `A_s` -- that name was
> already taken in this library by the Generalized Chaplygin Gas
> parameter, an unrelated quantity that shares the symbol in its own
> literature. Renaming the GCG one would invalidate every saved
> chain that names it.

### Six new datasets

- **DESI DR2 BAO (2025)**, `dataset_kwargs={"desi": {"version": "desi2025"}}` --
  three years of observations, >14 million galaxies and quasars,
  twice the DR1 sample, and the measurement the strengthened
  evolving-dark-energy claim rests on. Identical file format to DR1,
  so it drops into the same loader. **Not** to be stacked with DR1:
  DR2 contains every DR1 galaxy.
- **Union3** (`"union3"`) -- 2087 supernovae fit with the UNITY1.5
  hierarchical model and released as 22 binned distance moduli. The
  third of the three compilations the DESI dark-energy results are
  argued with, and they *disagree* about how far the data sits from
  a cosmological constant: DES-SN5YR pulls hardest, Pantheon+ least,
  Union3 in between. A library that can only fit one of them cannot
  reproduce that comparison, which is the actual state of the
  evidence.
- **Low-z BAO** (`"bao_lowz"`) -- 6dFGS (z = 0.106) and SDSS DR7 MGS
  (z = 0.15), the only BAO leverage below z = 0.2 (DESI starts at
  0.295, BOSS at 0.38). Independent of both, so unlike DESI-vs-SDSS
  these *can* be added to either. Two details are handled rather than
  papered over: 6dFGS reports `r_s/D_V`, kept as its own observable
  rather than inverted (inverting a Gaussian gives something that is
  neither Gaussian nor centred on 1/mean), and its measurement is
  calibrated against an Eisenstein-Hu fitting-formula sound horizon,
  so the theory `r_d` is rescaled by 153.9/149.8 -- 2.7% on a
  4.5%-precision point, the same class of definitional mismatch
  v0.19.0 documents.
- **`"h0"`, `"omega_b"`, `"tau"`** -- external single-number
  measurements: SH0ES 2022/2024 and TDCOSMO 2025 time-delay lensing
  for H0, BBN (Schoeneberg 2024 or Cooke 2018) for omega_b h^2, and
  Planck lowE for tau.

  These are **datasets, not priors**, and the distinction is
  deliberate. The posterior is identical either way, but a prior is a
  statement of belief before seeing data, while "SH0ES measured
  73.04 +- 1.04" is data from a telescope with a systematic error
  budget. As a dataset it shows up in the per-dataset chi2
  breakdown, in the degrees-of-freedom count AIC/BIC use, in figure
  legends, and in the chain metadata that decides whether a saved
  chain may be resumed. A fit that quietly assumed the local
  distance ladder should not look, from the outside, like a fit that
  did not. It also makes the H0 tension askable in the form it is
  argued: run the same model with and without `"h0"` and compare what
  each dataset contributes.

The BBN prior is more than an extra data point: BAO measures
`D/r_d`, and with a BBN constraint on `omega_b` a BAO-only fit gains
a CMB-independent route to H0 -- exactly how the DESI "BAO + BBN"
constraints are produced.

### Overlapping datasets now warn

Every "don't combine X and Y" rule in this README was, until now,
written down only in a docstring, where it protected nobody who did
not go looking. `Fitter` now checks and warns, naming the reason.
The failure it guards against is silent: overlapping data treated as
independent gives a perfectly ordinary-looking posterior with error
bars that are too small. It stays a warning, not an error --
quantifying how much an overlap matters is a legitimate thing to
want to do.

Two new rules join the existing ones: Union3 must not be combined
with Pantheon+ or DES-SN5YR, and `"planck"` must not be combined with
`"planck_lite"` (the distance priors are a compression of exactly
those bandpowers -- that is the entire CMB dataset twice).

### Eight new models

Grouped by what they actually change, since that decides what can be
done with them:

| Model | Extra parameters | Reduces to |
|---|---|---|
| **LogarithmicDE** | none (reuses `w0`, `wa`) | wCDM at `wa = 0` |
| **PEDE** | **none at all** | — |
| **GEDE** | `Delta`, `z_t` | LCDM at `Delta -> 0`; PEDE at `Delta = 1, z_t = 0` |
| **LsCDM** | `z_dagger` | LCDM below the transition |
| **IDE** | `xi` (with `w0`) | wCDM at `xi = 0` |
| **RunningVacuum** | `nu` | LCDM at `nu = 0` |
| **Cardassian** | `n_card`, `q_card` | LCDM at `n = 0, q = 1` |
| **DGP** | **none at all** | — |

Every reduction in that last column is a **test**
(`tests/test_models.py`), asserted to machine precision, not a
docstring claim. That matters more than it sounds: the Friedmann
closure `E(0) = 1` pins some normalizations but not all of them, and
a limit check pins the rest.

Two of these -- PEDE and DGP -- have **exactly LCDM's parameter
count** and a completely different expansion history. That makes the
model comparison unusually clean: identical AIC/BIC penalties, so a
chi2 difference is a difference in fit and nothing else.

DGP also overrides `mu(a, k)` with the standard Koyama-Maartens
result, so it joins the three modified-gravity models in predicting a
growth history that differs from GR's at fixed background. On the
self-accelerating branch gravity is *weaker* (`mu = 0.72` today,
matching the literature), and the bundled fsigma8 data feels it
directly.

### A matter-scaling bug this surfaced

RunningVacuum and IDE both change how *matter* dilutes -- that is
their whole content. But `GrowthCalculator` read `Omega_m(a)` off
`Omega_m (1+z)^3` for every model, so those two would have been given
LCDM's growth source term alongside their own `E(z)`: internally
inconsistent, and silent. `Cosmology` now has an overridable
`Omega_matter(z)` hook that both models implement and every other
model inherits unchanged. No existing result changes; the two new
models get a growth history that is actually theirs.

### The first test suite

The library had no tests. It now has **130**, running in ~10 s, and
they are aimed at the failure mode this project actually has:
physics that is *plausibly* wrong rather than broken.

- `test_models.py` -- Friedmann closure for every model (flat and
  curved), `dEdz` against a central finite difference (two
  independent hand transcriptions of the same algebra), every known
  limit, published signatures (PEDE's `w(0) = -1.145`, DGP's
  `Omega_rc` and growth suppression), and a bound on the error the
  distance integrator's grid makes on LsCDM's genuine discontinuity.
- `test_datasets.py` -- every dataset loads, every likelihood's
  covariance matches its data vector, and every chi2 per data point
  is O(1) at a concordance cosmology. The two deliberate exceptions
  are the tensions themselves: `"h0"` disagrees at ~5 sigma and
  `"s8"` at ~2.9 sigma, and the bounds are set just above those so a
  *further* error would still be caught rather than hidden.
- `test_planck_lite.py` -- the binning and covariance algebra against
  published log-likelihoods, and the CAMB translation separately
  against physics, so a failure says which half is wrong.

Run them with `pip install -e ".[dev]" && pytest`.

## v0.22.0

A figure-typography pass: everything a plot
says is now written the way a paper would write it, and a few labels
that were quietly *wrong* are fixed.

Parameter names reach every axis and corner-plot title as LaTeX
(`$\Omega_b$`, `$r_d$`, `$\sigma_8$`) instead of as Python
identifiers -- the labels were always declared on the parameter
container, the plots just weren't reading them. Model names do the
same: legends show `ΛCDM`, `wCDM`, `f(R)` Hu-Sawicki via a new
`Cosmology.MODEL_LABEL` / `plot_label()` (with `plain_name()` for
tables, dropdowns and JSON, where raw LaTeX would be shown
literally), and `define_model(..., label=...)` lets a custom model
supply its own. Corner-plot titles now size their precision per
parameter, so `Omega_b` reads `0.0491 +0.0011 -0.0011` instead of
corner's default `0.05 +0.00 -0.00`.

Three labels were misleading rather than merely plain:

- **The Pantheon+ Hubble diagram's y axis claimed to be
  `mu = m_B - M_B`.** It is neither: the plotted data is the
  corrected *apparent* magnitude `m_b_corr` (~11-27 mag, not a
  distance modulus's ~33-46), and `M_B` is analytically
  marginalized out by default, so it cannot appear in an axis
  label. Now `$m_B$ [mag]`, with the model curve's legend entry
  saying where its normalization came from
  (`Model ($M_B$ marginalized)`).
- **The BAO panels were titled `D_V/r_s` while their y axis said
  `r_d`.** Both denote the sound horizon at the drag epoch; the
  `rs` spelling comes from DESI's own data file, which
  `MODEL_MAP`'s keys keep, but the code divides by `rd` and now so
  do the titles.
- **The Planck pull plot's ticks were the raw dataset identifiers**
  (`lA`, `omega_b_h2`); they now render as `$\ell_A$`,
  `$\omega_b h^2$`.

Dataset combinations in legends also spell themselves out
(`CC + DESI + Pantheon+`, via `stats.dataset_label`) rather than
joining registry keys. No numerical result changes from any of this --
it is labels, titles and legends only.

The DES-SN5YR Hubble diagram is also legible for the first time. That
release ships 81 supernovae (of 1820) with `mu_err` between 5 and 468
mag -- entries the survey de-weights rather than removes -- and
matplotlib autoscales an errorbar plot to contain every whisker, so
the panel spanned +/-500 mag with the actual 35-45 mag Hubble diagram
compressed into a sliver of it. Panels are now scaled by the
measurements rather than by their largest error bars, and a bar may
only stretch the view if it is extreme both as a fraction of the
data's spread *and* as a multiple of that dataset's median error --
which caps nothing at all on CC, DESI, Pantheon+ or fsigma8, where
matplotlib's own limits are reproduced exactly. The 81 points stay on
the plot, drawn without the whiskers that no longer fit and counted
in the legend.

One functional bug surfaced while testing the above and is fixed here
too: **`FitResult.save_json()` (and the GUI's JSON download) failed
for any fit that used `run_mcmc(save=...)`**, with
`TypeError: Object of type int64 is not JSON serializable`. Reading a
chain's step count back through an HDF5 attribute yields `np.int64`
rather than `int`, and `json` rejects it (`np.float64` slips through
only because it subclasses `float`). The counters are now cast at the
source, and both JSON writers coerce numpy scalars rather than
failing on a save.

## v0.21.0

Adds the w0-wa dark-energy plane
(`fitter.plots.w0_wa_plane()`, and `compare_w0_wa_plane()` for
several posteriors at once): 2D credible contours over the
phantom/quintessence/quintom-A/quintom-B regions with ΛCDM marked at
(-1, 0) -- the figure the DESI evolving-dark-energy results are
argued in. The classification behind it is available on its own as
`cpl_diagnostics.classify_region()` /
`cpl_diagnostics.region_fractions()`, which turns "the contours sit
in the quintom-B region" into a posterior probability. Contour levels
are stated as 2D credible probabilities (68%/95% of the samples), not
sigmas, since in two dimensions the familiar 1D contours enclose only
39%/86.5%.

The same release fixes a reporting bug in the GUI's w0-wa
diagnostics: it printed the Mahalanobis distance D of the ΛCDM point
with a σ suffix. In 2D, D is not a number of sigma (D² follows χ² with
2 degrees of freedom), so this overstated the tension -- D = 2.20
reads as "2.2σ" but is really 1.70σ. The library was corrected in
v0.19.0; the GUI now reports `sigma` (with the confidence level, and D
labelled as what it is) too.

## v0.20.0

Makes MCMC chains persistent. Until now a chain
lived only in memory: closing the notebook, or adding one more plot
to the bottom of a script, meant sampling the whole posterior again
from step zero -- hours, to recompute samples that hadn't changed.
`run_mcmc(save="chains/fit.h5")` now writes the chain to HDF5 as it
is sampled (emcee's own `HDFBackend`, plus a `cosmofit` metadata
group), and picks it back up on the next call instead of re-running
it. `nsteps` counts the *total* length to reach, so re-running an
unchanged script samples nothing, raising `nsteps` samples only the
difference, and an interrupted run keeps every step it had already
taken. `Fitter.from_chain("chains/fit.h5")` reopens a finished run in
a later session -- model, datasets, free parameters, priors and fixed
values all come back out of the file -- and
`CosmoFit.stats.chains.open_chain()` reads the posterior with no
dataset loaded at all.

Resuming is exact, not approximate: emcee's proposal RNG state is
stored with the walkers, so a chain sampled in four sittings is
bit-identical to the same chain sampled in one (verified directly, as
is multi-core `n_processes` writing through the same file). The
metadata is also what keeps it *safe* -- a resume whose model,
datasets, free parameters, prior bounds or fixed parameter values
differ from the stored chain's is refused, naming exactly what
differs, rather than silently welding samples from two different
posteriors into one array. The GUI uses the same machinery
(`fitter.chain_id()`, a stable hash of the posterior, as the
filename), so adding a model to a comparison no longer re-runs the
models already fitted. `h5py` is now a dependency.

## v0.19.0

Fixes a real scientific error in the Planck
distance-prior likelihood. Evaluated at *Planck's own best-fit
LCDM* -- where a correct implementation must return chi2 ~ 0 -- the
old code returned **chi2 ~ 100 for 3 data points**, with `l_A` off by
**-8.9 sigma** (`R` and `omega_b_h2` were fine at 0.13 and 0.07
sigma, which is what localized it to the sound horizon `r_s(z*)`).

The cause was not a bug in the physics but a **definitional
mismatch**, which is the classic trap with compressed likelihoods.
These priors are not a measurement of the sky; they are a summary of
Planck's own fit, computed by Chen, Huang & Wang (2019) under a
specific set of conventions. CosmoFit was computing a *more detailed*
prediction than the compression assumed -- radiation as photons plus
3.046 massless neutrinos (`omega_r = 4.18e-5`) where CHW19 define
`Omega_r = Omega_m/(1+z_eq)` (0.8% lower, massive neutrinos left in
`Omega_m`), and `z*` from the Hu & Sugiyama (1996) fitting formula
where CHW19 take it from the Planck chains, i.e. from CAMB. HS96 was
calibrated against 1990s recombination physics and runs 0.22% high
for Planck-like parameters (1091.9 vs CAMB's 1089.9); `l_A` is
sensitive enough to `z*` that this alone is a ~4 sigma shift. Being
*more* physical than the data's own definitions is still wrong when
the data is a compression.

`RecombinationCalculator` now follows CHW19 Eqs. (1)-(6) exactly, and
`z*` comes from a fit calibrated directly against **CAMB 2.0.1** over
a 12x14 grid in (`omega_b`, `omega_cb`) covering far more than the
Planck posterior, accurate to **0.0018%** in `z*` (~0.04 sigma of the
`l_A` prior). The radiation term is also renormalized properly:
adding `Omega_r(1+z)^4` on top of a model's `E(z)` left
`E(0)^2 = 1 + Omega_r`, over-closing the universe; the correction
reuses each model's own dark-energy evolution, so it stays exact for
curved and non-LCDM models alike. The same fiducial check now gives
`l_A` +0.54 sigma, `R` -0.03 sigma, **chi2 = 0.39**.

Cross-checked two independent ways: against CAMB (`z*` to 0.002%),
and against a separate `scipy.quad` implementation of the CHW19
recipe across flat/open/closed LCDM and CPL (`R` and `l_A` to
<0.01 sigma). Per-evaluation cost is unchanged.

**This changes results.** The bias did not show up as a bad fit --
the sampler absorbed it by shifting parameters, which is exactly what
makes this kind of error dangerous. Refitting CC+DESI+Pantheon+ +
Planck:

| | old | new |
|---|---|---|
| LCDM `H0` | 68.04 | **67.39** |
| LCDM `Omega_m` | 0.3118 | **0.3149** |
| CPL `w0` | -0.973 | **-0.881** |
| CPL `wa` | -0.000 | **-0.298** |

The CPL case is the headline: the old code put `wa` at essentially
zero -- perfectly consistent with a cosmological constant -- while
the corrected code prefers evolving dark energy, in the same
direction and of comparable size to DESI's own published w0waCDM
result. Any CPL/JBP/BA conclusion drawn from a Planck-including fit
made with v0.18.0 or earlier should be regenerated.

> **Note:** don't combine two different `"s8"` versions in the same
> fit (default is `"kids1000"`; pass
> `dataset_kwargs={"s8": {"version": "des_y3"}}` for the other) --
> they're independent survey constraints, not a joint one. See the
> `S8Likelihood` docstring.

## v0.17.0

Adds growth-of-structure: the "phase 2" v0.16.0
explicitly deferred, since a background-only treatment left
FRHuSawicki's `f_R0`/`n` genuinely untestable (its background *is*
LCDM's by construction). A new `cosmology.calculators.growth.GrowthCalculator`
solves the standard sub-horizon, quasi-static linear growth ODE
(`D'' + (2+dlnH/dN)D' - (3/2)Omega_m(a) mu(a,k) D = 0`) for every
model via a `Cosmology.mu(a,k)` hook (default 1, i.e. standard GR
growth -- LCDM/wCDM/CPL/JBP/BA/GCG all get this for free), exposing
`fitter.cosmology.background.{growth_rate,sigma8,fsigma8}(z)`, plus
a new `sigma8` cosmological parameter and two new datasets/likelihoods:
`"fsigma8"` (the "Gold-2018" RSD growth-rate compilation, Sagredo,
Nesseris & Sapone 2018, arXiv:1806.10822, 22 points, with the same
Alcock-Paczynski correction the reference likelihood applies) and
`"s8"` (a single Gaussian S8 = sigma8*sqrt(Omega_m/0.3) constraint,
KiDS-1000 or DES Y3). Each of the three modified-gravity models now
has a real, cited `mu(a,k)`: **FQExponential** uses the settled
sub-horizon result G_eff/G_N = 1/f_Q (Barros, Barreiro, Koivisto &
Nunes 2020, arXiv:2004.07867); **FRTLinear** uses
`mu = 1 + 3*beta` (the same coupling already in its own `E(z)^2`,
stated explicitly as a simplification rather than a full covariant
perturbation derivation -- see the class docstring); and
**FRHuSawicki** gets the standard chameleon-screened, scale- and
time-dependent `mu(a,k)` (Pogosian & Silvestri 2008, arXiv:0709.0296),
derived here directly from this model's own background rather than
transcribed, held at a fixed fiducial pivot `k=0.1` h/Mpc since
CosmoFit's fsigma8 data are single per-z points, not a P(k) shape --
`f_R0`/`n` are still inert for background-only fits, but now
genuinely shape `"fsigma8"`/`"s8"` predictions (verified end-to-end:
a short FRHuSawicki fit against `"fsigma8"` alone now gives `f_R0` a
real, visibly informative posterior, not a flat one spanning the
whole prior). `fitter.plots.growth()`/`compare_growth()` add the
fsigma8(z) diagram alongside the existing figures, and the GUI picks
up both new datasets/the new plot automatically through the same
`DATASET_REGISTRY`/`EXTRA_PARAMS` mechanism every other dataset/model
already goes through.

## v0.16.0

Adds modified-gravity models -- **FQExponential**
(f(Q) gravity), **FRTLinear** (f(R,T) gravity), and **FRHuSawicki**
(f(R) gravity) -- a real category beyond dark-energy-on-top-of-GR
reparametrizations (LCDM/wCDM/CPL/JBP/BA) or a unified fluid (GCG):
these modify the gravitational field equations themselves.
**Background (E(z)) level only** -- CosmoFit's datasets (CC, BAO,
SNe, Planck distance priors) are all background/expansion-history
probes, and growth-of-structure data (fσ8/RSD) plus the perturbation
machinery to use it isn't implemented, so that's the honest ceiling
of what's testable here for now. Both FQExponential's and FRTLinear's
Friedmann equations were derived and verified directly against
primary sources (Anagnostopoulos, Basilakos & Saridakis 2021,
arXiv:2104.15123, for f(Q); Harko, Lobo, Nojiri & Odintsov 2011,
arXiv:1104.2669, for f(R,T)) rather than taken from a single
secondary source -- one transcription (a sign error in FQExponential's
Lambert-W closure formula) was caught this way, by a numerical
closure self-check (`E(z=0)` should equal 1 by construction; it
didn't, until the sign was fixed) that's now a permanent part of the
model's test coverage. **FRHuSawicki's background is identical to
LCDM's by construction** -- the standard "designer f(R)" approach
builds f(R) to reproduce an assumed target background, so `f_R0`/`n`
are present as parameters but don't affect `E(z)` at all; fitting
them against these datasets won't meaningfully constrain them. This
is stated explicitly in the class docstring and shown as a visible
warning next to the model picker in the GUI -- included for
completeness rather than silently omitted, but not silently
overstated either. All three plug into the existing `EXTRA_PARAMS`
mechanism (built for `define_model()`/custom models), so `Fitter`,
every plot including `compare_*`, and the GUI's multi-model
comparison all work with them with no further changes.

## v0.15.0

Fixes `n_processes` on Python 3.14: it changed
`multiprocessing`'s default start method on Linux from `fork` to
`forkserver` (matching what macOS/Windows already used), which made
`run_mcmc(n_processes=...)` crash outright as a plain script
(`forkserver`/`spawn` require every process-spawning call to sit
behind an `if __name__ == "__main__":` guard) and, even inside a
Jupyter kernel where that particular crash doesn't surface, measured
no speedup at all. `Fitter._mcmc_pool` now explicitly requests the
`fork` context rather than relying on whatever the interpreter's
default happens to be -- still available on Linux/macOS even where
it's no longer automatic, and with neither of the above problems.
This version also brings the graphical interface up to parity with
`cpl_mcmc_tfd42.ipynb`: the GUI can now configure and fit multiple
models at once (built-in or custom) sharing one set of datasets, with
a statistical comparison tab (AIC/BIC, and a likelihood-ratio test
when exactly two models are properly nested), every `compare_*`
figure alongside the existing single-model ones, and the CPL-family
w(z)=-1-crossing/LCDM-distance posterior diagnostics -- so everything
that notebook does is now also reachable with zero code. Every figure
also gets a download button (SVG/PNG/PDF, picked once per session and
applied to all of them) -- the browser's own save dialog is what lets
you choose where it goes.

**Resolved (v0.18.0): the "no speedup inside Jupyter" limitation was
a misdiagnosis.** Multiprocessing was never the problem, and it was
never specific to notebooks. The real cause was the *per-evaluation*
cost: every likelihood evaluation solved the Pantheon+ covariance
with a Cholesky triangular solve (`cho_solve`), and a triangular
solve is an inherently sequential recurrence -- each element depends
on the one before it -- so BLAS cannot thread it and worker processes
contend on memory bandwidth instead of scaling. The whole MCMC was
therefore pinned near one core's throughput in *every* environment;
a plain script only looked better because that is where
`n_processes` was actually being passed.

The covariance is constant, so `DenseCovariance` now precomputes an
explicit inverse once (validated against the original matrix, and
falling back to the Cholesky path if it fails that check) and
`solve()` is a symmetric mat-vec, which BLAS *does* thread. Measured
on the bundled 1624x1624 Pantheon+ covariance:

| | per solve | 8-process scaling |
|---|---|---|
| `cho_solve` (before) | 1.70 ms | 4.8x |
| mat-vec, 1 BLAS thread | 0.80 ms | 7.5x |
| mat-vec, threaded BLAS | 0.18 ms | -- |

End result for a 3-dataset CPL chain on an 8-core machine: the
single-process path went from 1.1x to **7.9x** core utilization and
roughly halved in wall time, *without any multiprocessing at all* --
and it measures the same in a plain script, in Jupyter Lab, in VS
Code, and under `nbconvert`. chi2 is unchanged to ~1e-11 relative.

`n_processes` now also defaults to `"auto"`, so notebooks get
multi-core behaviour without passing anything: it uses every core the
process is *allowed* to run on (`os.sched_getaffinity`, not
`os.cpu_count()` -- the two differ inside a container, cgroup, or
SLURM allocation, and oversizing the pool there makes things slower),
but only when the run is long enough to earn back worker startup, and
it silently stays single-process for a `define_model()` model rather
than raising. The chain is unaffected: the proposal RNG lives in the
main process, so a given `seed` gives bit-identical results at any
`n_processes` (verified).

## v0.14.0

Adds multi-core MCMC: `fitter.run_mcmc(n_processes=...)`
evaluates the ensemble's walkers across multiple CPU cores. This
isn't emcee's own naive recipe (pairing a raw `multiprocessing.Pool`
with `Fitter`'s already-built, data-carrying log-posterior) -- for a
dataset like Pantheon+ (a ~1600x1600 dense covariance matrix),
re-pickling and sending that to a worker on every single step
measured *slower* than one process (~19ms to pickle vs. ~2ms to
evaluate). Instead, each worker process builds its own `Fitter` once
(from a small, cheap-to-pickle recipe), via the pool's `initializer`,
and only a length-`ndim` float vector crosses the process boundary
per evaluation after that; workers also pin their own BLAS thread
pool to 1 (via the new `threadpoolctl` dependency) to avoid
oversubscribing the machine. Net effect, measured on an 8-core
machine for a 4-dataset CPL fit: ~2.4x, well under `n_processes`x
(per-step IPC has its own cost) but a real, significant speedup. Only
works for models picklable by reference (every built-in model; not a
dynamically-built `define_model()`/`model_from_expression()` model,
which raises a clear error rather than an obscure pickling failure
if `n_processes` is given), and is most reliable on Linux/macOS
(multiprocessing from a Windows notebook is fragile for reasons
outside this library's control). The new `examples/05-case-studies/cpl_mcmc_tfd42.ipynb`
notebook is a publication-scale variant of `cpl_mcmc_analysis.ipynb`
using this: all four datasets (including Planck, which makes `rd`
and `Omega_b` constrainable and lets them join the free parameters
too) and a much longer chain (`nwalkers=64`, `nsteps=12000`), run in
parallel across every available core.

## v0.13.0

Adds model comparison plots: every
`fitter.plots.*` figure (`hz`, `deceleration`, `w_of_z`,
`hubble_diagram`, `des_hubble_diagram`, `bao_distances`,
`sdss_bao_distances`) gets a `compare_*` counterpart that overlays
this fit's curve with one or more other models' curves on the same
data/axes -- the "model A vs model B" figures cosmology papers use,
rather than only being able to look at models one at a time or
compare them statistically (AIC/BIC/LRT) without a picture of what
the difference actually looks like. `other_fits=None` (the default)
auto-compares against a quick best-fit-only LCDM reference built
from the same datasets; passing an already-fit `Fitter`, or a list
of them, compares against exactly those instead, for an arbitrary
N-model figure. Every `compare_*` method reuses the corresponding
single-model method's existing evaluation logic (posterior-predictive
bands, per-fit analytic SN offsets, ...) rather than duplicating it,
and works for any model -- built-in or
[custom](#custom-models) -- since it only depends on
`Fitter`/`Cosmology`'s existing generic interface. The
`cpl_mcmc_analysis` notebook's CPL vs. LCDM section now includes
these alongside its existing AIC/BIC/likelihood-ratio comparison.

## v0.12.0

Adds a graphical interface: a
[Streamlit](https://streamlit.io) app (`app/streamlit_app.py`,
optional `pip install -e ".[gui]"`) for ticking datasets, picking a
built-in model or writing a custom `E(z)` as an expression, editing
free parameters/bounds in a table, and running the fit + rendering
plots with one click -- a pure UI layer over the existing
`Fitter`/`FitPlotter` API, with no changes to either. The one new
library addition is `model_from_expression()`
(`cosmology.custom`), a thin wrapper around `define_model()` that
takes `E`/`w`/`dEdz` as expression strings (e.g.
`"sqrt(Omega_m*(1+z)**3 + ...)"`) instead of Python callables --
evaluated with builtins stripped and only whitelisted `numpy` math
plus the model's own parameters reachable, appropriate for a
locally-run tool.

## v0.11.0

Adds support for custom, not-in-the-literature
models: `define_model()` (`cosmology.custom`) builds a usable
`Cosmology` subclass from a single `E(z)` function -- which alone is
enough to fit against every built-in dataset/likelihood and produce
every plot except `w_of_z()`/`deceleration()` (an optional `dEdz=`
enables the latter, else a numerical finite-difference fallback is
used) -- plus any new parameters the model needs beyond the standard
set (`H0`, `Omega_m`, `Omega_k`, `w0`, `wa`, ...), each declared
inline with a default and prior bounds. The same mechanism is also
available by subclassing `Cosmology` directly and declaring an
`EXTRA_PARAMS` class attribute (`Cosmology.__init_subclass__` builds
a matching parameter dataclass and property automatically); this is
what `define_model` does under the hood. Required generalizing
`Fitter` to read the parameter container off the model
(`model.PARAMS_CLASS`) instead of hardcoding
`CosmologyParameters` -- every built-in model is unaffected (none
declare `EXTRA_PARAMS`), verified against all six.

## v0.10.0

Splits MCMC sampling out of `Fitter` and into a
dedicated `stats.sampler` module: `EnsembleSampler` (the existing
`emcee`-backed walker initialization/run logic, unchanged in
behavior) now implements a small `BaseSampler` interface, so the
sampling backend is a swappable component rather than logic
inlined in `Fitter.run_mcmc`, and `run_mcmc()` gains a `moves=`
argument to pass through custom `emcee` proposals (e.g.
`emcee.moves.DEMove()` for strongly correlated posteriors). It also
adds a consolidated result interface (`stats.results`): `fitter.result`
returns a `FitResult` bundling the best-fit point (`BestFitResult`)
and MCMC posterior summary/convergence (`MCMCResult`) that were
previously only available piecemeal via `best_fit_params`,
`best_fit_chi2`, `summary()`, `convergence()`, with a single
readable `repr` and `FitResult.save_json()` / `.load_json()` for
keeping a fit's headline numbers without pickling the `emcee`
sampler. Both are purely additive -- every existing `Fitter`
method/attribute (`run_mcmc()`, `best_fit()`, `summary()`,
`convergence()`, `samples_dict()`, `.sampler`, `.best_fit_result`,
...) is unchanged.

## v0.9.0

A performance pass, driven by profiling rather
than guesswork. The big one: evaluating `PlanckLikelihood` was
dominated by two integrals (the sound horizon and the comoving
distance to z\*) computed with `scipy.integrate.quad` -- an *adaptive
scalar* quadrature that calls the (already fully vectorized)
integrand at one z at a time, several hundred times per call. Both
are now evaluated on a fixed, vectorized grid (`scipy.integrate.simpson`)
instead -- one array-valued call instead of hundreds of scalar ones.
Getting this right took two tries: the first grid (linear in the
substituted variable) looked fine across randomized-parameter testing
but turned out to badly under-resolve the sound-horizon integral's
approach to its asymptote at realistic (Planck-fiducial-like)
parameter values specifically -- caught by comparing end-to-end
`PlanckLikelihood` output at a fixed point against the original
`quad`-based result, not just spot-checking the integral in
isolation. A log-spaced grid fixes it, verified to <1e-6 relative
error against `quad` across dozens of randomized and
literature-realistic parameter sets, across all 6 models. **Net
effect: ~8x faster evaluation of a joint CC+DESI+Pantheon+Planck
likelihood.** The distance-integrator's interpolation grid
(`cosmology.numerics.integrals`) was also shrunk several-fold
(verified: still >100x more accurate than any dataset's measurement
precision needs) for a smaller additional gain that applies to every
fit, Planck or not.

## v0.8.0

Adds the **SDSS BAO** likelihood: BOSS DR12
(z=0.38, 0.51) + eBOSS DR16 LRG (z=0.698) + eBOSS DR16 QSO (z=1.48),
combined into one dataset with a block-diagonal covariance (each
component is an independent, non-overlapping-redshift measurement).
Data downloaded directly from
[CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data)
(the same repository that is, independently, also the exact source of
this project's existing DESI 2024 files -- confirmed byte-for-byte)
and cross-checked against the published eBOSS DR16 LRG/QSO
uncertainties. BOSS DR12's usual third bin (z=0.61) is deliberately
omitted, since it overlaps the eBOSS DR16 LRG redshift range. The
DESI/SDSS-BAO `model()`/`residuals()`/`chi2()` logic (previously
duplicated in `DESILikelihood`) was factored into a shared
`BAODistanceLikelihood` base class.

> **Note:** don't combine `"desi"` and `"sdss_bao"` in the same fit --
> DESI targets much of the same sky BOSS/eBOSS did, so treating them
> as independent double-counts structure. See the
> `SDSSBAOLikelihood` docstring.

## v0.7.0

Adds the **DES-SN5YR** (Dark Energy Survey 5-year)
supernova likelihood, downloaded directly from the official
[des-science/DES-SN5YR](https://github.com/des-science/DES-SN5YR)
data release and cross-validated against its own reference likelihood
implementation (the distance-modulus formula and analytic
marginalization both match it exactly). Its covariance is shipped as a
precision (inverse covariance) matrix rather than a covariance matrix,
so this version also adds `PrecisionCovariance`, used directly (no
inversion round-trip) instead of forcing it through the existing
Cholesky-based `DenseCovariance`. The Pantheon+ and DES-SN5YR
marginalization logic (previously duplicated) was factored into a
shared `AnalyticOffsetMixin`.

> **Note:** don't combine `"pantheon"` and `"des_sn5yr"` in the same
> fit -- DES-SN5YR's low-z anchor sample (~11% of it) is also compiled
> into Pantheon+, so fitting both double-counts those supernovae. See
> the `DESSN5YRLikelihood` docstring.

## v0.6.0

Adds three dark-energy models: **JBP** and **BA**
(alternative w0-wa parametrizations to CPL, reusing the same `w0`/`wa`
parameters) and **GCG** (Generalized Chaplygin Gas, a genuinely
different unified dark-matter/dark-energy fluid, adding two new shared
parameters `A_s`/`alpha`). All three have closed-form `E(z)`/`dE/dz`
(no per-step numerical integration), verified against finite-difference
derivatives and independent numerical integration of the continuity
equation, so they're exactly as fast in an MCMC as LCDM/wCDM/CPL.

## v0.5.0

Moves the package to a standard `src` layout
(`src/CosmoFit/...`) with a unified public API, so `from CosmoFit
import CPL, Fitter, ...` now works instead of importing the five
subpackages (`cosmology`, `data`, `likelihoods`, `stats`, `plots`)
separately from the repository root.

## v0.4.0

Fixes a Pantheon+ distance-modulus bug (the
zHD/zHEL redshift distinction from Brout et al. 2022 was not
applied) and a circular import between `data.loader` and
`likelihoods`, roughly halves the per-step cost of an MCMC run
that includes Pantheon+, adds the `wCDM` model, adds an
autocorrelation-time MCMC convergence check
(`Fitter.convergence()`), and moves all plotting into a dedicated
`plots.FitPlotter` (`fitter.plots`), with new Hubble diagram, H(z),
BAO distance, Planck pull, w(z) evolution and deceleration-parameter
figures alongside the existing chain/corner plots.

## v0.3.0

Fixes a curvature bug in the LCDM/CPL Friedmann
equations and in the transverse comoving distance (Omega_k was
previously a fittable-but-inert parameter), and adds a Planck 2018
CMB distance-prior likelihood (shift parameter R, acoustic scale
l_A, omega_b_h2), backed by a radiation-aware sound-horizon
calculation and the Hu & Sugiyama (1996) z_star fitting formula.

# Contributing

Thanks for looking. This file is short and mostly about two things
that are unusual here: how the code is formatted, and what counts as
a finished change.

## Getting set up

```bash
git clone https://github.com/salihyesil59/CosmoFit.git
cd CosmoFit
pip install -e ".[dev,cmb,theory,evidence,speed,docs]"
pytest -q
```

The suite takes about four minutes and everything must pass. All five
extras are optional at runtime, and the test suite skips what is not
installed -- but a change is not tested until it has run with them
present.

Work on `dev`. `main` is deliberately held at the last stable release.

## Formatting: there is no formatter, on purpose

`black` and `ruff format` would rewrite 101 of the 117 files under
`src/` and `tests/`, and there is nothing wrong with those 101 files. The code
is hand-wrapped: numbers aligned into columns, comments set against
the expression they explain, blank lines inside long formulae so the
terms of a Friedmann equation line up the way they do on paper. A
formatter flattens all of that, and takes every line of `git blame`
with it.

So match the surrounding code by eye. Concretely:

* four spaces, and lines that fit comfortably in about 72 columns of
  prose or 100 of code;
* a blank line between the parts of a long expression, where that is
  what makes it readable;
* numpydoc docstrings (`Parameters`, `Returns`, `Notes`,
  `References`) -- these are rendered by Sphinx, so a `Parameters`
  section holding a paragraph rather than parameters is a bug;
* equations in docstrings inside a `::` literal block. Indented plain
  text is a reStructuredText block quote, and a continuation line
  starting with `*`, `+` or `-` turns into a bullet list.

`ruff` runs as a **linter** and its findings do matter:

```bash
ruff check src tests app tools
```

## Annotations

The package ships `py.typed`, which tells every downstream type
checker to trust its annotations. That makes an annotation a promise
rather than a comment, and two tests hold the bar: every public
callable is fully annotated, and every annotation resolves. **Adding
a public method without annotating it fails the suite**, which is
deliberate.

Use the aliases in `CosmoFit.typing` for anything taking a redshift
and returning a number per redshift -- `z: Redshift` in and
`-> Array` out. `Array` is `np.float64 | NDArray[np.float64]` because
that is the truth: a scalar in gives a scalar out.

Where an annotation would need an import that is expensive or
circular, put the import under `if TYPE_CHECKING`. A checker reads
that block and the interpreter does not, which is the point --
`matplotlib.figure` costs 0.65 s to import, more than the rest of the
library put together, and somebody who never draws a figure should
not pay it.

`mypy` is in the `dev` extra:

```bash
mypy src/CosmoFit --ignore-missing-imports
```

It currently reports around eighty errors, all inside function bodies
rather than in the signatures `py.typed` is a promise about. CI does
not gate on it. Do not add to that number; reducing it is welcome.

## What a finished change looks like

**A test that fails before the fix.** For a bug, write the test
first and watch it fail; a test written afterwards has a habit of
passing for the wrong reason.

**Validation against something that does not share the machinery.**
This is the standard the library holds itself to and the reason to
trust any number it produces. A new model is checked against a
published constraint, or against a limit where it must reduce to one
already here. A new dataset reproduces the paper's own quoted
numbers, with those numbers used nowhere as an input. A new
calculation is checked against an independent route to the same
answer -- `quad` against a spline, one integrator against another,
the symbolic form against the textbook one. "It runs and the number
looks plausible" is where most of the bugs in the changelog got in.

**A commit message that says how you found out.** Read a few with
`git log`. When a bug produced a *plausible* answer -- which most of
the interesting ones did -- the story of how it surfaced is worth
more than the diff, because it is the part that stops the next one.

**The docs still build clean.**

```bash
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

`-W` is deliberate: warnings here are malformed docstrings that
render wrong on the page, and they only stay at zero if they fail.

## Data

Bundled datasets come from official public releases, unmodified, with
their provenance recorded in `REFERENCES.md`. Numbers are not
transcribed from memory, from a summary, or from a secondary source
-- a likelihood built on a misremembered covariance does not crash,
it just quietly answers a different question. If a dataset cannot be
obtained and checked against its own publication, it does not go in;
there are two entries in the Roadmap that say exactly this.

## Adding a model

The library has three routes that do not need a pull request at all,
and one of them is probably what you want:

| | |
|---|---|
| `define_model` | you have `E(z)` as a Python function |
| `model_from_expression` | you have `E(z)` as a string |
| `CosmoFit.theory.Action` | you have an *action*, and want the Friedmann equation derived from it |

See `examples/03-building-models/`. A model belongs in
`cosmology/models/` when it is established enough that somebody else
would come looking for it by name.

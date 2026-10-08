"""
Data too large to ship with the package, installed on demand.

Most of CosmoFit's data are a few megabytes and come with it. Some
likelihoods -- full CMB spectra, per-lens time-delay posteriors -- need
hundreds, and those are *data packages*: a URL, the SHA-256 the file
must have, and where it goes. ::

    cosmofit install NAME [NAME ...]      # by name
    cosmofit install input.yaml           # what an input's likelihoods need
    cosmofit install --list               # what can be installed, and what is

or from Python, :func:`install`. A likelihood names what it needs in its
``data_packages`` attribute and finds it with :func:`package_path`,
which says how to install it when it is missing.

Where they go: ``--path`` (``path=``), else the environment variable
``COSMOFIT_PACKAGES_PATH``, else ``~/.cosmofit/packages``; each package
in a directory of its own name.

A download is written next to its destination and checked before
anything is replaced, so an interrupted or corrupted download leaves
the previous installation -- or nothing -- rather than half a package.
Archives (``.zip``, ``.tar``, ``.tar.gz``, ``.tgz``) are unpacked, and a
member that would land outside the package's directory is refused.
Anything else is kept as the single file it is.

Packages besides the built-in ones register under the
``cosmofit.data_packages`` entry point, each a :class:`DataPackage` or
a list of them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile

from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable, Iterable


__all__ = [
    "DataPackage",
    "DataNotInstalledError",
    "InstallError",
    "available_packages",
    "data_path",
    "install",
    "is_installed",
    "package_path",
    "packages_for",
]


#: The environment variable that sets where packages go.
ENVIRONMENT = "COSMOFIT_PACKAGES_PATH"

#: Written into each installed package's directory: what was installed.
MARKER = ".cosmofit-package.json"

_CHUNK = 1 << 20


class InstallError(RuntimeError):
    """A download or an unpacking that failed, and was not installed."""


class DataNotInstalledError(FileNotFoundError):
    """A data package a likelihood needs is not installed."""


@dataclass(frozen=True)
class DataPackage:
    """
    One installable file.

    Attributes
    ----------
    name : str
        Its name, and its directory's.
    url : str
        Where it is downloaded from (``https://``, or ``file://``).
    sha256 : str
        The SHA-256 of the file at ``url``, hex. A download that does
        not have it is not installed.
    description, reference : str
        What it is, and whom to cite for it.
    size : int, optional
        Its size in bytes, for the listing.
    """

    name: str
    url: str
    sha256: str
    description: str = ""
    reference: str = ""
    size: int | None = None

    @property
    def filename(self) -> str:
        return self.url.rstrip("/").rsplit("/", 1)[-1]

    @property
    def archive(self) -> str | None:

        name = self.filename.lower()

        if name.endswith(".zip"):
            return "zip"

        if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            return "tar"

        return None


#: The packages that come with CosmoFit, by name. Filled in as the
#: likelihoods that need them are added.
PACKAGES: dict[str, DataPackage] = {}


def available_packages() -> dict[str, DataPackage]:
    """Every package that can be installed: built in, then registered."""

    found = dict(PACKAGES)

    for point in entry_points(group="cosmofit.data_packages"):

        loaded = point.load()

        for package in (loaded if isinstance(loaded, (list, tuple)) else [loaded]):

            if not isinstance(package, DataPackage):
                raise TypeError(
                    f"Entry point {point.name!r} of cosmofit.data_packages gave "
                    f"{type(package).__name__}, not a DataPackage."
                )

            found.setdefault(package.name, package)

    return found


def _package(name: str) -> DataPackage:

    packages = available_packages()

    if name not in packages:
        raise KeyError(
            f"No data package is called {name!r}; 'cosmofit install --list' "
            f"names them. Known: {sorted(packages) or 'none'}."
        )

    return packages[name]


# ============================================================
# Where
# ============================================================

def data_path(path: str | os.PathLike | None = None) -> Path:
    """
    Where packages are installed: ``path``, else ``$COSMOFIT_PACKAGES_PATH``,
    else ``~/.cosmofit/packages``.
    """

    if path is None:
        path = os.environ.get(ENVIRONMENT) or Path.home() / ".cosmofit" / "packages"

    return Path(path).expanduser()


def _marker(name: str, path=None) -> dict | None:

    marker = data_path(path) / name / MARKER

    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_installed(name: str, path=None) -> bool:
    """Whether ``name`` is installed -- the version the registry names now."""

    marker = _marker(name, path)

    return marker is not None and marker.get("sha256") == _package(name).sha256


def package_path(name: str, path=None) -> Path:
    """
    The directory of an installed package. Raises
    :class:`DataNotInstalledError`, saying how to install it, when it
    is not there -- or is an older version than the registry's.
    """

    package = _package(name)

    if not is_installed(name, path):

        where = data_path(path)

        old = _marker(name, path) is not None

        raise DataNotInstalledError(
            f"The data package {name!r} ({package.description or package.filename}) "
            f"is {'out of date' if old else 'not installed'} in {where}. "
            f"Install it with:\n\n    cosmofit install {name}\n\n"
            f"(or cosmofit.install.install([{name!r}])); set {ENVIRONMENT} to "
            f"keep packages elsewhere."
        )

    return data_path(path) / name


# ============================================================
# Install
# ============================================================

def _download(package: DataPackage, target: Path, report: Callable) -> None:
    """Stream ``package.url`` to ``target``, checking its SHA-256."""

    digest = hashlib.sha256()
    received = 0

    try:

        with urllib.request.urlopen(package.url, timeout=60) as response, \
                open(target, "wb") as out:

            total = response.headers.get("Content-Length")
            total = int(total) if total else package.size

            while chunk := response.read(_CHUNK):

                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)

        report(
            f"  {package.name}: downloaded {received / 1e6:.1f} MB"
            + (f" of {total / 1e6:.1f}" if total and total != received else "")
        )

    except OSError as error:
        raise InstallError(f"{package.name}: could not download {package.url}: {error}") from error

    if digest.hexdigest() != package.sha256.lower():
        raise InstallError(
            f"{package.name}: {package.url} has SHA-256 {digest.hexdigest()}, not the "
            f"{package.sha256} it is registered with. Nothing was installed: the "
            f"file changed upstream, or the download was corrupted."
        )


def _inside(root: Path, member: str) -> bool:

    target = (root / member).resolve()

    return target == root or root in target.parents


def _unpack(package: DataPackage, archive: Path, destination: Path) -> None:

    destination.mkdir(parents=True)
    root = destination.resolve()

    if package.archive is None:
        shutil.copyfile(archive, destination / package.filename)
        return

    if package.archive == "zip":

        with zipfile.ZipFile(archive) as bundle:

            for member in bundle.namelist():
                if not _inside(root, member):
                    raise InstallError(
                        f"{package.name}: the archive's member {member!r} would be "
                        f"written outside {destination}; refused."
                    )

            bundle.extractall(destination)

        return

    with tarfile.open(archive) as bundle:

        for member in bundle.getmembers():

            if not _inside(root, member.name) or (
                (member.issym() or member.islnk())
                and not _inside(root, str(Path(member.name).parent / member.linkname))
            ):
                raise InstallError(
                    f"{package.name}: the archive's member {member.name!r} would be "
                    f"written outside {destination}; refused."
                )

        bundle.extractall(destination, **(
            {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        ))


def install(
    names: Iterable[str],
    path=None,
    force: bool = False,
    report: Callable[[str], None] = print,
) -> dict[str, Path]:
    """
    Install data packages; return each one's directory.

    A package already installed (at the version the registry names) is
    left alone unless ``force``. Every download is checked against its
    SHA-256 before it replaces anything.
    """

    root = data_path(path)
    root.mkdir(parents=True, exist_ok=True)

    installed = {}

    for name in dict.fromkeys(names):

        package = _package(name)
        destination = root / name

        if is_installed(name, path) and not force:
            report(f"  {name}: already installed in {destination}")
            installed[name] = destination
            continue

        report(f"  {name}: downloading {package.url}")

        download = root / f".{name}.download"
        staging = root / f".{name}.staging"

        for leftover in (download, staging):
            if leftover.is_dir():
                shutil.rmtree(leftover)
            elif leftover.exists():
                leftover.unlink()

        try:

            _download(package, download, report)
            _unpack(package, download, staging)

            (staging / MARKER).write_text(
                json.dumps(asdict(package), indent=2), encoding="utf-8",
            )

            if destination.exists():
                shutil.rmtree(destination)

            staging.rename(destination)

        except (InstallError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:

            if staging.exists():
                shutil.rmtree(staging)

            if isinstance(error, InstallError):
                raise

            raise InstallError(f"{name}: could not unpack {package.filename}: {error}") from error

        finally:

            if download.exists():
                download.unlink()

        report(f"  {name}: installed in {destination}")
        installed[name] = destination

    return installed


# ============================================================
# What an input needs
# ============================================================

def packages_for(info: dict) -> list[str]:
    """
    The data packages an input's components need: each likelihood's and
    theory's ``data_packages`` attribute, in order, without repeats.
    """

    from cosmofit.core.registry import resolve

    needed = []

    for kind, block in (("theory", "theory"), ("likelihood", "likelihood")):

        for name, options in (info.get(block) or {}).items():

            cls = resolve(kind, name, options or {})

            needed.extend(getattr(cls, "data_packages", ()))

    return list(dict.fromkeys(needed))

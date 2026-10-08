"""
``cosmofit install``: data packages downloaded, checked and unpacked.

Every download here is a ``file://`` URL into a temporary directory, so
nothing touches the network -- and nothing the user has installed.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile

import pytest

from cosmofit import install as data
from cosmofit.cli import main
from cosmofit.core import Likelihood


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def packages(tmp_path, monkeypatch):
    """
    Three packages -- a zip, a tar.gz and a single file -- registered
    for the test, and an install directory of its own.
    """

    source = tmp_path / "source"
    source.mkdir()

    zipped = source / "spectra.zip"
    with zipfile.ZipFile(zipped, "w") as bundle:
        bundle.writestr("spectra/cl.txt", "2 1.0\n3 0.5\n")
        bundle.writestr("spectra/README", "test")

    tarred = source / "lenses.tar.gz"
    with tarfile.open(tarred, "w:gz") as bundle:
        payload = b"0.1 0.2\n"
        member = tarfile.TarInfo("lenses/posterior.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    single = source / "prior.txt"
    single.write_text("70.0 1.0\n")

    registry = {
        path.stem.split(".")[0]: data.DataPackage(
            name=path.stem.split(".")[0], url=path.as_uri(), sha256=sha256(path),
            description=f"test {path.name}", size=path.stat().st_size,
        )
        for path in (zipped, tarred, single)
    }

    monkeypatch.setattr(data, "PACKAGES", registry)

    root = tmp_path / "installed"
    monkeypatch.setenv(data.ENVIRONMENT, str(root))

    return root


def test_where_packages_go(monkeypatch, tmp_path):

    monkeypatch.delenv(data.ENVIRONMENT, raising=False)
    assert data.data_path() == data.data_path(None)
    assert data.data_path().parts[-2:] == (".cosmofit", "packages")

    monkeypatch.setenv(data.ENVIRONMENT, str(tmp_path / "env"))
    assert data.data_path() == tmp_path / "env"

    assert data.data_path(tmp_path / "given") == tmp_path / "given"


def test_archives_and_files_are_installed(packages):

    paths = data.install(["spectra", "lenses", "prior"], report=lambda line: None)

    assert (paths["spectra"] / "spectra" / "cl.txt").read_text() == "2 1.0\n3 0.5\n"
    assert (paths["lenses"] / "lenses" / "posterior.txt").read_text() == "0.1 0.2\n"
    assert (paths["prior"] / "prior.txt").read_text() == "70.0 1.0\n"

    for name in paths:
        assert data.is_installed(name)
        assert data.package_path(name) == packages / name

    marker = json.loads((packages / "spectra" / data.MARKER).read_text())
    assert marker["sha256"] == data.PACKAGES["spectra"].sha256

    # Nothing left behind but the packages.
    assert sorted(p.name for p in packages.iterdir()) == ["lenses", "prior", "spectra"]


def test_an_installed_package_is_not_downloaded_again(packages):

    data.install(["prior"], report=lambda line: None)

    lines = []
    data.install(["prior"], report=lines.append)
    assert lines == [f"  prior: already installed in {packages / 'prior'}"]

    lines.clear()
    data.install(["prior"], force=True, report=lines.append)
    assert any("downloaded" in line for line in lines)


def test_a_wrong_checksum_installs_nothing(packages, monkeypatch):

    data.install(["prior"], report=lambda line: None)

    good = data.PACKAGES["prior"]
    monkeypatch.setitem(data.PACKAGES, "prior", data.DataPackage(
        name="prior", url=good.url, sha256="0" * 64,
    ))

    with pytest.raises(data.InstallError, match="SHA-256"):
        data.install(["prior"], force=True, report=lambda line: None)

    # The earlier installation is untouched, and nothing half-done is left.
    assert (packages / "prior" / "prior.txt").read_text() == "70.0 1.0\n"
    assert sorted(p.name for p in packages.iterdir()) == ["prior"]

    # ...and no longer counts as installed: it is not what the registry names.
    with pytest.raises(data.DataNotInstalledError, match="out of date"):
        data.package_path("prior")


@pytest.mark.parametrize("member", ["../escaped.txt", "/absolute.txt"])
def test_archive_members_cannot_escape(packages, monkeypatch, tmp_path, member):

    evil = tmp_path / "source" / "evil.zip"

    with zipfile.ZipFile(evil, "w") as bundle:
        bundle.writestr(member, "x")

    monkeypatch.setitem(data.PACKAGES, "evil", data.DataPackage(
        name="evil", url=evil.as_uri(), sha256=sha256(evil),
    ))

    with pytest.raises(data.InstallError, match="outside"):
        data.install(["evil"], report=lambda line: None)

    assert not (packages / "evil").exists()
    assert not (packages.parent / "escaped.txt").exists()


def test_tar_links_cannot_escape(packages, monkeypatch, tmp_path):

    evil = tmp_path / "source" / "link.tar"

    with tarfile.open(evil, "w") as bundle:
        link = tarfile.TarInfo("inside/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        bundle.addfile(link)

    monkeypatch.setitem(data.PACKAGES, "link", data.DataPackage(
        name="link", url=evil.as_uri(), sha256=sha256(evil),
    ))

    with pytest.raises(data.InstallError, match="outside"):
        data.install(["link"], report=lambda line: None)


def test_a_missing_package_says_how_to_install_it(packages):

    with pytest.raises(data.DataNotInstalledError, match="cosmofit install lenses"):
        data.package_path("lenses")

    with pytest.raises(KeyError, match="No data package is called"):
        data.package_path("nothing")


def test_an_unreachable_url(packages, monkeypatch, tmp_path):

    monkeypatch.setitem(data.PACKAGES, "gone", data.DataPackage(
        name="gone", url=(tmp_path / "missing.zip").as_uri(), sha256="0" * 64,
    ))

    with pytest.raises(data.InstallError, match="could not download"):
        data.install(["gone"], report=lambda line: None)


# ============================================================
# What an input needs, and the command
# ============================================================

class NeedsLenses(Likelihood):

    data_packages = ("lenses", "prior")

    def logp(self):
        return 0.0


def test_an_inputs_packages_come_from_its_components(packages):

    info = {
        "theory": {"background": None},
        "likelihood": {
            "mine": {"class": NeedsLenses},
            "also": {"class": NeedsLenses},
            "bao.desi": None,
        },
    }

    assert data.packages_for(info) == ["lenses", "prior"]


def test_the_command(packages, tmp_path, capsys):

    assert main(["install", "--list"]) == 0
    listing = capsys.readouterr().out
    assert "spectra" in listing and " installed " not in listing

    assert main(["install", "spectra", "prior"]) == 0
    assert data.is_installed("spectra") and data.is_installed("prior")
    capsys.readouterr()

    main(["install", "--list"])
    assert capsys.readouterr().out.count(" installed ") == 2

    elsewhere = tmp_path / "elsewhere"
    assert main(["install", "prior", "--path", str(elsewhere)]) == 0
    assert data.is_installed("prior", elsewhere)

    assert main(["install", "nothing"]) == 1
    assert "No data package is called 'nothing'" in capsys.readouterr().err


def test_the_command_installs_what_an_input_needs(packages, tmp_path, capsys):

    path = tmp_path / "input.yaml"
    path.write_text(
        "theory: {background: null}\n"
        "likelihood:\n"
        f"  lenses: {{class: {__name__}.NeedsLenses}}\n"
        "params: {H0: 70, Omega_m: 0.3, Omega_b: 0.05}\n"
    )

    assert main(["install", str(path)]) == 0

    assert "needs: lenses, prior" in capsys.readouterr().out
    assert data.is_installed("lenses") and data.is_installed("prior")

from automlcli import util


def test_ext_match() -> None:
    assert util.ext_match("foo/bar.csv", ["csv"])
    assert util.ext_match("foo/bar.csv.gz", ["csv"])
    assert not util.ext_match("foo/bar.tar.gz", ["csv"])

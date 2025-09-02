from toolchain import __version__


def test_version():
    assert __version__ is not None, "Version should not be None"
    assert isinstance(__version__, str), "Version should be a string"
    assert __version__ != "", "Version should not be an empty string"

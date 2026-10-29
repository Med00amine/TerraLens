import pytest

from terralens import __version__, cli


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommand_help_is_forwarded(capsys):
    with pytest.raises(SystemExit):
        cli.main(["train", "--help"])
    assert "--img-size" in capsys.readouterr().out


def test_unknown_command_fails():
    with pytest.raises(SystemExit) as exc:
        cli.main(["fly"])
    assert exc.value.code == 2

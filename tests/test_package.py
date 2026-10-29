import re

import terralens
from terralens.constants import CLASS_COLORS, CLASS_LABELS, CLASS_NAMES, color_for, label_for


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", terralens.__version__)


def test_every_class_has_label_and_color():
    assert len(CLASS_NAMES) == 10
    assert set(CLASS_LABELS) == set(CLASS_NAMES)
    assert set(CLASS_COLORS) == set(CLASS_NAMES)
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in CLASS_COLORS.values())


def test_unknown_class_falls_back():
    assert label_for("Glacier") == "Glacier"
    assert color_for("Glacier") == "#9e9e9e"

"""A layout model's picture box is accepted when it lies inside the image and covers most of it."""

import pytest

from tests.boxes import is_the_same_picture

# (layout box, the image PyMuPDF reports) for the five figures on the sample's first four pages,
# as the accurate service returned them. The layout model crops to the visible figure, so its
# box is often tighter than the image object, which carries white margin (figure A: 84% of it).
MEASURED = {
    "page 1": ([297.2, 63.4, 535.7, 242.7], [294.9, 64.7, 539.8, 244.5]),
    "page 2": ([44.4, 64.2, 270.1, 192.6], [43.8, 60.2, 272.9, 196.3]),
    "page 3 figure A": ([55.3, 65.0, 258.1, 209.0], [41.9, 61.8, 273.7, 211.6]),
    "page 3 figure B": ([51.3, 241.9, 259.8, 384.4], [48.7, 241.4, 261.9, 384.9]),
    "page 3 figure C": ([300.7, 62.6, 534.0, 175.4], [301.3, 62.2, 533.5, 177.6]),
}
REAL = [41.9, 61.8, 273.7, 211.6]


@pytest.mark.parametrize("name", MEASURED)
def test_the_boxes_the_service_returns_are_accepted(name):
    layout, real = MEASURED[name]

    assert is_the_same_picture(layout, real)


@pytest.mark.parametrize(
    "wrong",
    [
        pytest.param([x * 2 for x in REAL], id="pixels at twice the scale, not points"),
        pytest.param([REAL[1], REAL[0], REAL[3], REAL[2]], id="x and y swapped"),
        pytest.param([0.0, 0.0, 612.0, 792.0], id="the whole page"),
        pytest.param([100.0, 100.0, 120.0, 110.0], id="a sliver inside the image"),
        pytest.param([300.0, 300.0, 500.0, 450.0], id="a different picture"),
    ],
)
def test_a_box_that_is_wrong_is_rejected(wrong):
    assert not is_the_same_picture(wrong, REAL)

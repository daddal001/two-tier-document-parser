"""Compare a layout model's box for a picture with where PyMuPDF says the image really sits."""


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def is_the_same_picture(layout, real, *, inside=0.95, covers=0.75):
    """True when ``layout`` lies inside ``real`` and covers most of it (boxes are [x0, y0, x1, y1]).

    A layout model crops to the visible figure, so its box is often tighter than the image
    object, which can carry white margin. Intersection over union penalises that margin and so
    rejected a correct box (84% of its image). What matters is that the box is in page points and
    on the right picture: that holds when almost all of it lies inside the real image (a box in
    pixels, with swapped axes, or for another picture does not) and it is not a mere sliver of it.
    """
    shared = _intersection(layout, real)
    if _area(layout) == 0 or _area(real) == 0:
        return False
    return shared / _area(layout) >= inside and shared / _area(real) >= covers

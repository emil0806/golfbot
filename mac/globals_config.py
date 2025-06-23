import math
##### BORDER #####

FIELD_X_MIN = None
FIELD_X_MAX = None
FIELD_Y_MIN = None
FIELD_Y_MAX = None

PIX_FIELD_CORNERS = [
    (305,  65),     # TL
    (1635, 65),     # TR
    (1635, 1050),   # BR
    (305,  1050)    # BL
]

FIELD_LINES = []

def set_field_lines_from_corners():

    tl, tr, br, bl = PIX_FIELD_CORNERS
    lines = [
        (tl[0], tl[1], tr[0], tr[1]),  # top
        (tr[0], tr[1], br[0], br[1]),  # right
        (br[0], br[1], bl[0], bl[1]),  # bottom
        (bl[0], bl[1], tl[0], tl[1])   # left
    ]

    global FIELD_LINES
    FIELD_LINES = lines

def get_field_lines():
    return FIELD_LINES

def set_field_bounds_by_corners():

    xs = [pt[0] for pt in PIX_FIELD_CORNERS]
    ys = [pt[1] for pt in PIX_FIELD_CORNERS]

    global FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX
    FIELD_X_MIN = min(xs)
    FIELD_X_MAX = max(xs)
    FIELD_Y_MIN = min(ys)
    FIELD_Y_MAX = max(ys)


def get_field_bounds():
    return FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX

##### CROSS #####
CROSS_X_MIN = None
CROSS_X_MAX = None
CROSS_Y_MIN = None
CROSS_Y_MAX = None
CROSS_CENTER = None

CROSS_LINES = []

def set_cross_lines(lines):
    global CROSS_LINES
    CROSS_LINES = lines

def get_cross_lines():
    return CROSS_LINES

def set_cross_bounds(bounds, center):
    global CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, CROSS_CENTER
    CROSS_X_MIN = bounds["x_min"]
    CROSS_X_MAX = bounds["x_max"]
    CROSS_Y_MIN = bounds["y_min"]
    CROSS_Y_MAX = bounds["y_max"]
    CROSS_CENTER = center

def get_cross_bounds():
    return (CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX)

def get_cross_center():
    return CROSS_CENTER

def extract_cross_lines(detected_lines):
    if not detected_lines:
        g.set_cross_lines([])
        return

    horizontal_lines = []
    vertical_lines = []

    for x1, y1, x2, y2 in detected_lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):  # Horisontale linjer
            horizontal_lines.append((x1, y1, x2, y2))
        else:  # Vertikale linjer
            vertical_lines.append((x1, y1, x2, y2))

    def average_line(lines):
        if not lines:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs += [x1, x2]
            ys += [y1, y2]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    horizontal_lines.sort(key=lambda l: (l[1] + l[3]) // 2)
    vertical_lines.sort(key=lambda l: (l[0] + l[2]) // 2)

    top_line = horizontal_lines[0] if horizontal_lines else None
    bottom_line = horizontal_lines[-1] if len(horizontal_lines) > 1 else top_line
    left_line = vertical_lines[0] if vertical_lines else None
    right_line = vertical_lines[-1] if len(vertical_lines) > 1 else left_line

    mid_horizontal = average_line([top_line, bottom_line]) if top_line else None
    mid_vertical = average_line([left_line, right_line]) if left_line else None

    cross_lines = []
    if mid_horizontal:
        cross_lines.append(mid_horizontal)
    if mid_vertical:
        cross_lines.append(mid_vertical)

    set_cross_lines(cross_lines)

    # Sæt bounds + center til brug i pathfinding
    if mid_horizontal and mid_vertical:
        x_min = min(mid_vertical[0], mid_vertical[2])
        x_max = max(mid_vertical[0], mid_vertical[2])
        y_min = min(mid_horizontal[1], mid_horizontal[3])
        y_max = max(mid_horizontal[1], mid_horizontal[3])
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        set_cross_bounds({"x_min": x_min, "x_max": x_max,
                            "y_min": y_min, "y_max": y_max}, center)